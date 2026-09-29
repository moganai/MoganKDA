"""Tests for the per-channel-gate ``kkt_solve_kda`` (Akk^-1 and Aqk of a 64-token chunk).

Reference: fp64 torch, direct (un-anchored) Gamma_ij = sum_c k_ic k_jc 2^(g_ic - g_jc), Akk via
``solve_triangular``, Aqk; fed with the same fp32 chunk-local log2 cumsum ``g`` the kernel receives
(produced by ``chunk_local_cumsum(..., scale=1/ln2)``). Error metric: max|d| / max|ref| per output.
Also: FLA ``chunk_kda_fwd_intra`` as second reference (info + loose threshold), bitwise prefix invariance,
document isolation, determinism, timing.
"""
import pytest
import torch
import torch.nn.functional as F

from mogan_kda.ops.kimi_delta_rule.chunk.hopper.kkt_solve_kda import kkt_solve_kda
from mogan_kda.ops.utils import chunk_local_cumsum

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]

DEVICE = "cuda"
CHUNK = 64
RCP_LN2 = 1.4426950408889634
TOL_AQK, TOL_AKK, TOL_FLA = 1e-2, 2e-2, 5e-2
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs 1, 97, 256, 150, 200


def make_inputs(B, T, HK, HV, seed=0, gate="real"):
    """q, k L2-normalized bf16; per-token natural-log gate g_tok in [-5, 0); beta in (0, 2)."""
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEVICE, dtype=torch.float32)  # noqa: E731
    q = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    k = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    if gate == "real":
        g_tok = -5 * torch.sigmoid(r(B, T, HV, 128))
    else:  # "min": maximum decay everywhere, "zero": no decay
        g_tok = torch.full((B, T, HV, 128), -5.0 if gate == "min" else 0.0, device=DEVICE)
    beta = 2 * torch.sigmoid(r(B, T, HV))
    return q, k, g_tok, beta


def cu_tensor(cu_list):
    return None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)


def cumsum_log2(g_tok, cu):
    return chunk_local_cumsum(g_tok, chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2)


def run_kernel(q, k, g_tok, beta, cu):
    g = cumsum_log2(g_tok, cu)
    return g, *kkt_solve_kda(q, k, g, beta, cu_seqlens=cu)


def ref_kkt(q, k, g, beta, scale, cu_list):
    """fp64 reference; returns (A, Aqk) as [B, T, HV, 64]."""
    B, T, HK, K = q.shape
    HV = g.shape[2]
    q, k, g, beta = q.double(), k.double(), g.double(), beta.double()
    qx, kx = q.repeat_interleave(HV // HK, 2), k.repeat_interleave(HV // HK, 2)
    A = torch.zeros(B, T, HV, CHUNK, dtype=torch.float64, device=q.device)
    Aqk = torch.zeros_like(A)
    segs = [(b, 0, T) for b in range(B)] if cu_list is None else \
        [(0, cu_list[i], cu_list[i + 1]) for i in range(len(cu_list) - 1)]
    for b, s, e in segs:
        for c in range(s, e, CHUNK):
            L = min(CHUNK, e - c)
            kc, qc, gc, bc = kx[b, c:c + L], qx[b, c:c + L], g[b, c:c + L], beta[b, c:c + L]
            E = torch.exp2(gc[:, None] - gc[None])  # [L, L, HV, K]; only i >= j is used
            gamma = torch.einsum("ihc,jhc,ijhc->hij", kc, kc, E)
            aq = torch.einsum("ihc,jhc,ijhc->hij", qc, kc, E) * scale
            eye = torch.eye(L, dtype=torch.float64, device=q.device)
            M = eye + torch.tril(bc.T[:, :, None] * gamma, -1)
            inv = torch.linalg.solve_triangular(M, eye.expand(HV, L, L), upper=False)
            A[b, c:c + L, :, :L] = inv.permute(1, 0, 2)
            Aqk[b, c:c + L, :, :L] = torch.tril(aq).permute(1, 0, 2)
    return A, Aqk


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def lower_masks(dev):
    i = torch.arange(CHUNK, device=dev)
    return i[:, None] >= i[None, :], i[:, None] > i[None, :]  # incl. diagonal (Aqk), strict


def chunk_rows(T, cu_list):
    """Chunk-local row index (t - chunk start) of every token position."""
    lens = [T] if cu_list is None else [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]
    return torch.cat([torch.arange(n, device=DEVICE) % CHUNK for n in lens])


def masked_rel(a, b, mask, cu_list=None):
    """rel error over the lower-triangular chunk region (FLA leaves the rest of Aqk unwritten)."""
    full = mask[chunk_rows(a.shape[1], cu_list)][None, :, None, :].expand_as(a)
    return rel(a[full], b[full])


CONFIGS = [
    pytest.param(2, 300, 2, 4, None, id="dense-B2-T300-Hk2Hv4"),
    pytest.param(1, 704, 2, 4, VARLEN_CU, id="varlen-Hk2Hv4"),
    pytest.param(2, 130, 4, 4, None, id="dense-Hk4Hv4"),
]


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
@pytest.mark.parametrize("gate", ["real", "min", "zero"])
def test_accuracy_vs_fp64_reference(B, T, HK, HV, cu_list, gate):
    q, k, g_tok, beta = make_inputs(B, T, HK, HV, gate=gate)
    g, A, Aqk = run_kernel(q, k, g_tok, beta, cu_tensor(cu_list))
    assert torch.isfinite(A.float()).all() and torch.isfinite(Aqk.float()).all(), "non-finite output"
    A_ref, Aqk_ref = ref_kkt(q, k, g, beta, 128 ** -0.5, cu_list)
    e_akk, e_aqk = rel(A, A_ref), rel(Aqk, Aqk_ref)
    print(f"[{gate}] Akk rel err {e_akk:.3e} (tol {TOL_AKK:.0e}), Aqk rel err {e_aqk:.3e} (tol {TOL_AQK:.0e})")
    assert e_aqk < TOL_AQK and e_akk < TOL_AKK
    # both outputs are exactly zero above the diagonal
    above = (torch.arange(CHUNK, device=DEVICE)[None, :] > chunk_rows(T, cu_list)[:, None])[None, :, None, :]
    above = above.expand_as(A)
    assert (A[above] == 0).all() and (Aqk[above] == 0).all()


@pytest.mark.parametrize("safe_gate", [False, True], ids=["fla-default", "fla-safe_gate"])
def test_against_fla_chunk_kda_fwd_intra(safe_gate):
    """Second reference (dense, chunk aligned rows): FLA's Akk / Aqk on the lower-triangular region."""
    try:
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    except ImportError as e:
        pytest.skip(f"fla not importable: {e}")
    q, k, g_tok, beta = make_inputs(2, 256, 2, 4)
    g, A, Aqk = run_kernel(q, k, g_tok, beta, None)
    v = torch.randn(2, 256, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    _, _, _, _, aqk_f, akk_f = chunk_kda_fwd_intra(q, k, v, gk=g, beta=beta, scale=128 ** -0.5, safe_gate=safe_gate)
    with_diag, strict = lower_masks(DEVICE)
    A_ref, Aqk_ref = ref_kkt(q, k, g, beta, 128 ** -0.5, None)
    e_aqk, e_akk = masked_rel(Aqk, aqk_f, with_diag), masked_rel(A, akk_f, with_diag)
    print(f"ours vs FLA(safe_gate={safe_gate}): Aqk {e_aqk:.3e}, Akk {e_akk:.3e};  "
          f"FLA vs fp64 ref: Aqk {masked_rel(aqk_f, Aqk_ref, with_diag):.3e}, Akk {masked_rel(akk_f, A_ref, with_diag):.3e}")
    assert e_aqk < TOL_FLA and e_akk < TOL_FLA


PREFIX_POS = [1, 8, 9, 15, 16, 17, 63, 65, 193]


@pytest.mark.parametrize("P", PREFIX_POS)
def test_prefix_invariance_bitwise(P):
    """Perturbing tokens >= P (q, k, per-token g, beta) must not change rows < P of A and Aqk (bitwise)."""
    B, T, HK, HV = 1, 256, 2, 4
    q, k, g_tok, beta = make_inputs(B, T, HK, HV, seed=0)
    q2, k2, g2, b2 = make_inputs(B, T, HK, HV, seed=1)
    splice = lambda a, b: torch.cat([a[:, :P], b[:, P:]], 1)  # noqa: E731
    _, A0, Q0 = run_kernel(q, k, g_tok, beta, None)
    _, A1, Q1 = run_kernel(splice(q, q2), splice(k, k2), splice(g_tok, g2), splice(beta, b2), None)
    assert torch.equal(A0[:, :P], A1[:, :P]), f"Akk leaks future tokens (P={P})"
    assert torch.equal(Q0[:, :P], Q1[:, :P]), f"Aqk leaks future tokens (P={P})"


@pytest.mark.parametrize("doc", [0, 2, 4])
def test_varlen_document_isolation_bitwise(doc):
    B, T, HK, HV = 1, 704, 2, 4
    q, k, g_tok, beta = make_inputs(B, T, HK, HV, seed=0)
    q2, k2, g2, b2 = make_inputs(B, T, HK, HV, seed=1)
    s, e = VARLEN_CU[doc], VARLEN_CU[doc + 1]

    def splice(a, b):
        out = a.clone()
        out[:, s:e] = b[:, s:e]
        return out

    cu = cu_tensor(VARLEN_CU)
    _, A0, Q0 = run_kernel(q, k, g_tok, beta, cu)
    _, A1, Q1 = run_kernel(splice(q, q2), splice(k, k2), splice(g_tok, g2), splice(beta, b2), cu)
    keep = torch.ones(T, dtype=torch.bool, device=DEVICE)
    keep[s:e] = False
    assert torch.equal(A0[:, keep], A1[:, keep]) and torch.equal(Q0[:, keep], Q1[:, keep])
    # doc 0 has length 1 (A row is always [1, 0, ...]); Aqk[i, i] = scale * q.k still changes
    assert not (torch.equal(A0[:, ~keep], A1[:, ~keep]) and torch.equal(Q0[:, ~keep], Q1[:, ~keep])), \
        "perturbation had no effect (test is vacuous)"


def test_deterministic():
    q, k, g_tok, beta = make_inputs(1, 704, 2, 4)
    cu = cu_tensor(VARLEN_CU)
    _, A0, Q0 = run_kernel(q, k, g_tok, beta, cu)
    for _ in range(5):
        _, A1, Q1 = run_kernel(q, k, g_tok, beta, cu)
        assert torch.equal(A0, A1) and torch.equal(Q0, Q1)


def test_check_lower_bound_option():
    q, k, g_tok, beta = make_inputs(1, 128, 2, 4)
    g = cumsum_log2(g_tok, None)
    kkt_solve_kda(q, k, g, beta, check_lower_bound=True)  # per-token decay >= -5: passes
    bad = cumsum_log2(g_tok * 2.0, None)  # per-token decay down to -10
    with pytest.raises(AssertionError):
        kkt_solve_kda(q, k, bad, beta, check_lower_bound=True)


def _time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def test_timing():
    """Informational: [1, 16384, HK=8, HV=16, 128]. FLA's chunk_kda_fwd_intra also runs recompute_w_u."""
    q, k, g_tok, beta = make_inputs(1, 16384, 8, 16)
    g = cumsum_log2(g_tok, None)
    print(f"cumsum (vector, log2): {_time(lambda: cumsum_log2(g_tok, None)):.1f} us")
    print(f"kkt_solve_kda:          {_time(lambda: kkt_solve_kda(q, k, g, beta)):.1f} us")
    try:
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    except ImportError:
        return
    v = torch.randn(1, 16384, 16, 128, device=DEVICE, dtype=torch.bfloat16)
    for sg in (False, True):
        us = _time(lambda: chunk_kda_fwd_intra(q, k, v, gk=g, beta=beta, scale=128 ** -0.5, safe_gate=sg))
        print(f"FLA chunk_kda_fwd_intra (safe_gate={sg}, incl. recompute_w_u): {us:.1f} us")
