"""Fused gate activation + chunk-local cumsum (fwd / bwd), ``mogan_kda/ops/kimi_delta_rule/gate.py``.

Reference: fp64 ``ref_kda.ref_gate`` (same formulas as ``_log_decay``) + fp64 chunk-local cumsum; gradients by torch
autograd through ref_gate. Tolerance: |kernel - ref| <= REL * max|ref| (fp32 kernel, fp64 reference).
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_kda import ref_gate  # noqa: E402

from mogan_kda.ops.utils import chunk_local_cumsum  # noqa: E402
from mogan_kda.ops.kimi_delta_rule.gate import kda_gate_cumsum_fwd, kda_gate_bwd  # noqa: E402

DEVICE = "cuda"
CHUNK = 64
RCP_LN2 = 1.0 / math.log(2.0)
LB = -5.0
FWD_REL, DX_REL, RED_REL = 2e-5, 1e-4, 5e-4
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs 1, 97, 256, 150, 200
MODES = ["eda", "kimi", "gdn"]
CONFIGS = [
    pytest.param((2, 300, 4, 128), None, id="dense-B2-T300-H4"),
    pytest.param((2, 300, 16, 128), None, id="dense-B2-T300-H16"),
    pytest.param((1, 704, 4, 128), VARLEN_CU, id="varlen-H4"),
    pytest.param((1, 704, 16, 128), VARLEN_CU, id="varlen-H16"),
]


def make_inputs(shape, mode, dtype, seed=0):
    torch.manual_seed(seed)
    B, Tn, HV, K = shape
    x = (torch.randn(shape, device=DEVICE) * 1.5).to(dtype)
    # A_log = log(U(1, 16)): for "gdn" many tokens sit below the bound (clamp region), some above
    A_log = torch.log(torch.empty(HV, device=DEVICE).uniform_(1.0, 16.0))
    dt_bias = torch.randn(HV, K, device=DEVICE) * 0.5
    return x, A_log, dt_bias


def ref_chunk_cumsum(g, cu, reverse=False):
    out = torch.zeros_like(g)
    segs = [(b, 0, g.shape[1]) for b in range(g.shape[0])] if cu is None else \
        [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    for b, s, e in segs:
        for c in range(s, e, CHUNK):
            xx = g[b, c:min(c + CHUNK, e)]
            out[b, c:min(c + CHUNK, e)] = xx.flip(0).cumsum(0).flip(0) if reverse else xx.cumsum(0)
    return out


def to_cu(cu_list):
    return None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)


def n_valid(shape, cu_list):
    return shape[1] if cu_list is None else cu_list[-1]


def check(name, got, ref, rel):
    err = (got.double() - ref).abs().max().item()
    tol = rel * max(ref.abs().max().item(), 1e-3)
    assert err <= tol, f"{name}: max abs err {err:.3e} > {tol:.3e} (max|ref| {ref.abs().max().item():.3e})"


# ---------------------------------------------------------------------------
# forward values
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fwd_values(shape, cu_list, mode, dtype):
    x, A_log, dt_bias = make_inputs(shape, mode, dtype)
    cu = to_cu(cu_list)
    g_c = kda_gate_cumsum_fwd(x, mode, A_log, dt_bias, LB, cu)
    assert g_c.shape == x.shape and g_c.dtype == torch.float32
    n = n_valid(shape, cu_list)
    ref = ref_chunk_cumsum(ref_gate(x, mode, A_log, dt_bias, LB), cu_list) * RCP_LN2
    check("g_c", g_c[:, :n], ref[:, :n], FWD_REL)
    if mode == "gdn":  # the bound is enforced: per-token g >= l  <=>  first row of every chunk (= g_0 / ln2) >= l / ln2
        assert g_c[:, :n].min().item() >= LB * CHUNK * RCP_LN2 - 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS[:1] + CONFIGS[2:3])
def test_fwd_no_params(shape, cu_list):
    """kimi without A_log (E = 1) and dt_bias."""
    x, _, _ = make_inputs(shape, "kimi", torch.float32)
    cu = to_cu(cu_list)
    n = n_valid(shape, cu_list)
    g_c = kda_gate_cumsum_fwd(x, "kimi", None, None, LB, cu)
    ref = ref_chunk_cumsum(ref_gate(x, "kimi", None, None, LB), cu_list) * RCP_LN2
    check("g_c", g_c[:, :n], ref[:, :n], FWD_REL)


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_precomputed_bitwise(shape, cu_list, dtype):
    x, _, _ = make_inputs(shape, "precomputed", dtype)
    x = (-5 * torch.sigmoid(x.float())).to(dtype)  # gate-like values
    cu = to_cu(cu_list)
    n = n_valid(shape, cu_list)
    got = kda_gate_cumsum_fwd(x, "precomputed", None, None, None, cu)
    want = chunk_local_cumsum(x.float().contiguous(), chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2)
    assert torch.equal(got[:, :n], want[:, :n]), "precomputed path is not bitwise equal to chunk_local_cumsum"


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------

def ref_grads(x, mode, A_log, dt_bias, cu_list, dg):
    """fp64 autograd of sum(ref_gate(x) * dg) -> (dx, dA_log, ddt_bias)."""
    xr = x.double().requires_grad_(True)
    Ar = A_log.double().requires_grad_(True)
    br = dt_bias.double().requires_grad_(True)
    g = ref_gate(xr, mode, Ar, br, LB)
    n = n_valid(x.shape, cu_list)
    (g[:, :n] * dg[:, :n].double()).sum().backward()
    return xr.grad, Ar.grad, br.grad


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("prime", [False, True], ids=["dg", "dg_prime_fold_K5"])
def test_bwd_values(shape, cu_list, mode, dtype, prime):
    x, A_log, dt_bias = make_inputs(shape, mode, dtype)
    cu = to_cu(cu_list)
    n = n_valid(shape, cu_list)
    torch.manual_seed(1)
    dgp = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    if prime:  # dg = chunk-local reverse cumsum of dg'
        dg_ref = ref_chunk_cumsum(dgp.double(), cu_list, reverse=True)
    else:
        dg_ref = dgp.double()
    rdx, rdA, rdb = ref_grads(x, mode, A_log, dt_bias, cu_list, dg_ref)
    dx, dA, db = kda_gate_bwd(x, dgp, mode, A_log, dt_bias, LB, cu, dg_is_prime=prime)
    assert dx.dtype == dtype and dx.shape == x.shape
    assert dA.shape == A_log.shape and db.shape == dt_bias.shape
    check("dx", dx[:, :n].float(), rdx[:, :n], DX_REL if dtype == torch.float32 else 1e-2)
    check("dA_log", dA, rdA, RED_REL)
    check("ddt_bias", db, rdb, RED_REL)


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS[:1] + CONFIGS[2:3])
def test_bwd_gdn_clamp_region(shape, cu_list):
    """Where the clamp is active (-E softplus(s) < l) dx must be exactly 0; elsewhere it must be nonzero."""
    x, A_log, dt_bias = make_inputs(shape, "gdn", torch.float32)
    cu = to_cu(cu_list)
    n = n_valid(shape, cu_list)
    dg = torch.ones(shape, device=DEVICE, dtype=torch.float32)
    dx, _, _ = kda_gate_bwd(x, dg, "gdn", A_log, dt_bias, LB, cu)
    g0 = -A_log.double().exp().view(1, 1, -1, 1) * torch.nn.functional.softplus(x.double() + dt_bias.double())
    clamped = (g0 < LB - 1e-4)[:, :n]
    free = (g0 > LB + 1e-4)[:, :n]
    assert clamped.any() and free.any(), "test data must exercise both regions"
    assert (dx[:, :n][clamped] == 0).all(), "gradient must be exactly 0 in the clamp region"
    assert (dx[:, :n][free] != 0).all()


@pytest.mark.gpu
@pytest.mark.parametrize("shape, cu_list", CONFIGS[1:2] + CONFIGS[2:3])
@pytest.mark.parametrize("mode", MODES)
def test_determinism(shape, cu_list, mode):
    x, A_log, dt_bias = make_inputs(shape, mode, torch.float32)
    cu = to_cu(cu_list)
    n = n_valid(shape, cu_list)
    dg = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    f1 = kda_gate_cumsum_fwd(x, mode, A_log, dt_bias, LB, cu)
    f2 = kda_gate_cumsum_fwd(x, mode, A_log, dt_bias, LB, cu)
    assert torch.equal(f1[:, :n], f2[:, :n])
    r1 = kda_gate_bwd(x, dg, mode, A_log, dt_bias, LB, cu, dg_is_prime=True)
    r2 = kda_gate_bwd(x, dg, mode, A_log, dt_bias, LB, cu, dg_is_prime=True)
    assert torch.equal(r1[0][:, :n], r2[0][:, :n])
    assert torch.equal(r1[1], r2[1]) and torch.equal(r1[2], r2[2])


@pytest.mark.gpu
def test_prefix_invariance():
    """Chunk-local: the first 128 tokens of a T=300 run equal a T=128 run (bitwise)."""
    x, A_log, dt_bias = make_inputs((1, 300, 4, 128), "eda", torch.float32)
    a = kda_gate_cumsum_fwd(x, "eda", A_log, dt_bias, LB)
    b = kda_gate_cumsum_fwd(x[:, :128].contiguous(), "eda", A_log, dt_bias, LB)
    assert torch.equal(a[:, :128], b)


# ---------------------------------------------------------------------------
# timing (run with -s)
# ---------------------------------------------------------------------------

def _time(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "varlen16docs"])
def test_timing(mode, varlen):
    from mogan_kda.ops.kimi_delta_rule.chunk import _log_decay

    shape = (1, 16384, 16, 128)
    x, A_log, dt_bias = make_inputs(shape, mode, torch.float32)
    cu = torch.arange(0, 16385, 1024, device=DEVICE, dtype=torch.int32) if varlen else None
    dg = torch.randn(shape, device=DEVICE, dtype=torch.float32)

    def torch_fwd():
        return chunk_local_cumsum(_log_decay(x, mode, A_log, dt_bias, LB), chunk_size=CHUNK, cu_seqlens=cu,
                                  scale=RCP_LN2)

    def kern_fwd():
        return kda_gate_cumsum_fwd(x, mode, A_log, dt_bias, LB, cu)

    def torch_bwd():  # autograd through the torch gate + K5 reverse cumsum
        xr, Ar, br = (t.clone().requires_grad_(True) for t in (x, A_log, dt_bias))
        g = _log_decay(xr, mode, Ar, br, LB)
        gd = chunk_local_cumsum(dg, chunk_size=CHUNK, cu_seqlens=cu, reverse=True, zero_last_chunk=True)
        g.backward(gd)
        return xr.grad

    def kern_bwd():
        return kda_gate_bwd(x, dg, mode, A_log, dt_bias, LB, cu, dg_is_prime=True)

    tf, kf, tb, kb = _time(torch_fwd), _time(kern_fwd), _time(torch_bwd), _time(kern_bwd)
    print(f"\n[gate timing {mode} {'varlen' if varlen else 'dense'} {shape}] "
          f"fwd: torch {tf:.0f} us vs kernel {kf:.0f} us | bwd(+K5): torch {tb:.0f} us vs kernel {kb:.0f} us | "
          f"fwd+bwd: {tf + tb:.0f} vs {kf + kb:.0f} us")
