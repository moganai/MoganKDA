"""Chunk-local cumsum: per-channel ([B, T, H, K]) path and scalar ([B, T, H]) regression.

Reference: fp64 torch cumsum per (sequence, 64-token chunk), forward / reverse, optional scale.
Tolerance: the kernel accumulates in fp32 -> |kernel - ref| <= ATOL_REL * max|ref| (ATOL_REL = 5e-6,
about 40 fp32 ulps of the largest value, ample for a 64-element scan and tight enough to catch any
indexing / chunk-boundary error, which is O(1)).
"""
import pytest
import torch

from mogan_kda.ops.utils import chunk_local_cumsum

DEVICE = "cuda"
CHUNK = 64
RCP_LN2 = 1.4426950408889634
ATOL_REL = 5e-6
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs of length 1, 97, 256, 150, 200 (chunk-unaligned, incl. length 1)


def ref_cumsum(g, cu, reverse=False, scale=None):
    """fp64 chunk-local cumsum of g [B, T, ...]; sequences = batch rows (cu=None) or cu segments (B=1)."""
    g = g.double()
    out = torch.zeros_like(g)
    segs = [(b, 0, g.shape[1]) for b in range(g.shape[0])] if cu is None else \
        [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    for b, s, e in segs:
        for c in range(s, e, CHUNK):
            x = g[b, c:min(c + CHUNK, e)]
            out[b, c:min(c + CHUNK, e)] = x.flip(0).cumsum(0).flip(0) if reverse else x.cumsum(0)
    return out * scale if scale is not None else out


def make_g(shape):
    return -5 * torch.sigmoid(torch.randn(*shape, device=DEVICE, dtype=torch.float32))  # gate-like values in [-5, 0)


def check(g, cu_list, reverse, scale, zero_last_chunk=False):
    cu = None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)
    out = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, reverse=reverse,
                             zero_last_chunk=zero_last_chunk, scale=scale)
    assert out.shape == g.shape and out.dtype == g.dtype
    n = g.shape[1] if cu_list is None else cu_list[-1]
    ref = ref_cumsum(g, cu_list, reverse, scale)
    err = (out[:, :n].double() - ref[:, :n]).abs().max().item()
    assert err <= ATOL_REL * ref.abs().max().item(), f"max abs err {err:.3e} (max|ref| {ref.abs().max().item():.3f})"
    if zero_last_chunk:  # tokens past the last sequence must be zero
        assert out[:, n:].abs().max().item() == 0.0
    out2 = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, reverse=reverse,
                              zero_last_chunk=zero_last_chunk, scale=scale)
    assert torch.equal(out[:, :n], out2[:, :n]), "not bitwise reproducible"
    return out


# ---------------------------------------------------------------------------
# per-channel path
# ---------------------------------------------------------------------------

VECTOR_CONFIGS = [
    pytest.param((2, 300, 4, 128), None, id="dense-B2-T300-H4-K128"),
    pytest.param((1, 704, 16, 128), VARLEN_CU, id="varlen-H16-K128"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", VECTOR_CONFIGS)
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "rev"])
@pytest.mark.parametrize("scale", [None, RCP_LN2], ids=["noscale", "rcpln2"])
def test_vector_cumsum(shape, cu_list, reverse, scale):
    torch.manual_seed(0)
    check(make_g(shape), cu_list, reverse, scale)


@pytest.mark.gpu
def test_vector_many_heads_varlen():
    torch.manual_seed(0)
    check(make_g((1, 300, 32, 128)), [0, 1, 100, 300], False, None)


@pytest.mark.gpu
def test_vector_k64():
    torch.manual_seed(0)
    check(make_g((2, 130, 2, 64)), None, False, None)


@pytest.mark.gpu
def test_vector_zero_last_chunk():
    """Varlen input padded past cu[-1]: reverse (backward-gradient usage) with zero_last_chunk zeroes the padding."""
    torch.manual_seed(0)
    check(make_g((1, 704 + 37, 16, 128)), VARLEN_CU, True, None, zero_last_chunk=True)


@pytest.mark.gpu
def test_vector_equals_scalar_kernel_per_head():
    """A per-channel gate that is constant along K must reproduce the scalar path head by head."""
    torch.manual_seed(0)
    gs = make_g((1, 704, 16))
    vec = chunk_local_cumsum(gs[..., None].expand(-1, -1, -1, 128).contiguous(),
                             cu_seqlens=torch.tensor(VARLEN_CU, device=DEVICE, dtype=torch.int32))
    sca = chunk_local_cumsum(gs, cu_seqlens=torch.tensor(VARLEN_CU, device=DEVICE, dtype=torch.int32))
    assert torch.equal(vec[..., 0], sca) and torch.equal(vec[..., 127], sca)


# ---------------------------------------------------------------------------
# scalar path regression (behavior must be unchanged)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", [pytest.param((2, 300, 16), None, id="dense"),
                                            pytest.param((1, 704, 16), VARLEN_CU, id="varlen")])
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "rev"])
def test_scalar_cumsum(shape, cu_list, reverse):
    torch.manual_seed(0)
    check(make_g(shape), cu_list, reverse, None)


@pytest.mark.gpu
def test_scalar_zero_last_chunk():
    torch.manual_seed(0)
    check(make_g((1, 704 + 37, 16)), VARLEN_CU, True, None, zero_last_chunk=True)


@pytest.mark.gpu
def test_scalar_scale():
    torch.manual_seed(0)
    check(make_g((1, 704, 16)), VARLEN_CU, False, RCP_LN2)


# ---------------------------------------------------------------------------
# timing (informational)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_vector_cumsum_timing():
    g = make_g((1, 16384, 16, 128))
    for reverse in (False, True):
        for _ in range(10):
            chunk_local_cumsum(g, reverse=reverse)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            chunk_local_cumsum(g, reverse=reverse)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1e3 / 100
        gb = 2 * g.numel() * 4 / 1e9
        print(f"vector cumsum [1,16384,16,128] fp32 reverse={reverse}: {us:.1f} us  ({gb / (us * 1e-6):.0f} GB/s)")
