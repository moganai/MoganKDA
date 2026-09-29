"""Tests for K4 ``bwd_intra_kda`` (intra-chunk gradients through Gamma / Aqk).

Oracle: ``torch_ref.bwd_intra_torch`` (fp64, anchored=True) fed with the SAME bf16 dA / fp32 partial tensors. The kernel
anchors the column side at the first row of j's block (oracle: row block of i); both are mathematically identical,
so the comparison is at the metric level: max|d| / max|ref| per output < 5e-3 (values are printed).
"""
import pytest
import torch
import torch.nn.functional as F

from mogan_kda.ops.kimi_delta_rule.chunk import torch_ref as TR
from mogan_kda.ops.kimi_delta_rule.chunk.hopper import bwd_intra_kda

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]

DEVICE = "cuda"

F64 = torch.float64
TOL = 5e-3
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs 1, 97, 256, 150, 200
SCALE = 128 ** -0.5

CONFIGS = [
    pytest.param(2, 300, 2, 4, None, id="dense-B2-T300-Hk2Hv4"),
    pytest.param(1, 704, 2, 4, VARLEN_CU, id="varlen-Hk2Hv4"),
]


def make(B, T, HK, HV, seed=0):
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEVICE, dtype=torch.float32)  # noqa: E731
    q = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    k = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    g_tok = -5 * torch.sigmoid(r(B, T, HV, 128))  # per-token natural-log decay in [-5, 0)
    beta = 2 * torch.sigmoid(r(B, T, HV))
    return q, k, g_tok, beta


def cu_t(cu_list):
    return None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)


def chunk_cumsum_log2(g_tok, cu):
    B, T, HV, K = g_tok.shape
    g_c = torch.zeros_like(g_tok)
    _, chunks, _ = TR._chunk_list(B, T, cu)
    for ch in chunks:
        sl = slice(ch["s"], ch["s"] + ch["L"])
        g_c[ch["b"], sl] = (torch.cumsum(g_tok[ch["b"], sl].double(), 0) * TR.RCP_LN2).float()
    return g_c


def random_dA(B, T, HV, cu, seed=1, amp=0.05):
    torch.manual_seed(seed)
    dAqk = torch.zeros(B, T, HV, 64, device=DEVICE)
    dL = torch.zeros_like(dAqk)
    _, chunks, _ = TR._chunk_list(B, T, cu)
    for ch in chunks:
        b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
        m = amp * torch.randn(HV, L, L, device=DEVICE)
        dAqk[b, sl, :, :L] = torch.tril(m).permute(1, 0, 2)
        dL[b, sl, :, :L] = torch.tril(amp * torch.randn(HV, L, L, device=DEVICE), -1).permute(1, 0, 2)
    return dAqk.bfloat16(), dL.bfloat16()


def real_dA(q, k, g_tok, beta, cu, seed=2):
    """dAqk / dL / partials from the oracle's real forward + backward (fp64 on GPU), rounded like K3a will store them."""
    B, T, HK, _ = q.shape
    HV = g_tok.shape[2]
    torch.manual_seed(seed)
    v = torch.randn(B, T, HV, 128, device=DEVICE, dtype=F64)
    do = torch.randn(B, T, HV, 128, device=DEVICE, dtype=F64)
    f = TR.chunk_fwd_torch(q, k, v, g_tok, beta, SCALE, None, cu)
    r = TR.chunk_bwd_torch(q, k, v, beta, f["g_c"], f["A"], f["Aqk"], f["h"], do, None, SCALE, cu)
    it = r["inter"]
    return (it["dAqk"].bfloat16(), it["dL"].bfloat16(), it["dq_p"].float(), it["dk_p"].float(), it["dg_p"].float())


def oracle(q, k, g_c, beta, dAqk, dL, parts, cu):
    zero = torch.zeros(q.shape[0], q.shape[1], g_c.shape[2], 128, dtype=F64, device=DEVICE)
    p = [zero] * 3 if parts is None else [x.double() for x in parts]
    return TR.bwd_intra_torch(q.double(), k.double(), g_c.double(), beta.double(), dAqk.double(), dL.double(), *p,
                              SCALE, cu, anchored=True)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
@pytest.mark.parametrize("source", ["random", "real"])
@pytest.mark.parametrize("with_partials", [False, True], ids=["nopart", "part"])
def test_vs_oracle(B, T, HK, HV, cu_list, source, with_partials):
    q, k, g_tok, beta = make(B, T, HK, HV)
    cu = cu_t(cu_list)
    g_c = chunk_cumsum_log2(g_tok, cu)
    if source == "real":
        dAqk, dL, *parts = real_dA(q, k, g_tok, beta, cu)
    else:
        dAqk, dL = random_dA(B, T, HV, cu)
        torch.manual_seed(5)
        parts = [0.1 * torch.randn(B, T, HV, 128, device=DEVICE) for _ in range(3)]
    parts = parts if with_partials else None
    ref = oracle(q, k, g_c, beta, dAqk, dL, parts, cu)
    out = bwd_intra_kda(q, k, g_c, beta, dAqk, dL, *(parts or (None, None, None)), scale=SCALE, cu_seqlens=cu)
    errs = {n: rel(o, ref[key]) for n, o, key in zip(("dq", "dk", "dg", "dbeta"), out, ("dq", "dk", "dg", "dbeta_g"))}
    print(f"[{source}, partials={with_partials}] " + " ".join(f"{n} {e:.2e}" for n, e in errs.items()))
    for n, e in errs.items():
        assert e < TOL, (n, errs)


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
def test_diagonal_only_gives_exactly_zero_dg(B, T, HK, HV, cu_list):
    q, k, g_tok, beta = make(B, T, HK, HV)
    cu = cu_t(cu_list)
    g_c = chunk_cumsum_log2(g_tok, cu)
    torch.manual_seed(3)
    dAqk = torch.zeros(B, T, HV, 64, device=DEVICE)
    _, chunks, _ = TR._chunk_list(B, T, cu)
    for ch in chunks:
        for i in range(ch["L"]):
            dAqk[ch["b"], ch["s"] + i, :, i] = torch.randn(HV, device=DEVICE)
    dAqk = dAqk.bfloat16()
    dq, dk, dg, dbeta = bwd_intra_kda(q, k, g_c, beta, dAqk, torch.zeros_like(dAqk), scale=SCALE,
                                      cu_seqlens=cu)
    assert torch.equal(dg, torch.zeros_like(dg)), "diagonal dAqk leaked into dg'"
    assert torch.equal(dbeta, torch.zeros_like(dbeta))
    ref = oracle(q, k, g_c, beta, dAqk, torch.zeros_like(dAqk), None, cu)
    # dq / dk are bf16 outputs (rounding <= 2^-9 relative)
    assert rel(dq, ref["dq"]) < 4e-3 and rel(dk, ref["dk"]) < 4e-3


@pytest.mark.parametrize("P", [1, 15, 16, 17, 63, 65, 193])
def test_prefix_invariance_bitwise(P):
    """Perturb q, k, g, beta for rows >= P and zero dAqk / dL rows >= P: outputs of rows < P are bitwise equal."""
    B, T, HK, HV = 1, 256, 2, 4
    x, y = make(B, T, HK, HV, seed=0), make(B, T, HK, HV, seed=1)
    pert = tuple(torch.cat([a[:, :P], b[:, P:]], 1) for a, b in zip(x, y))
    dAqk, dL = random_dA(B, T, HV, None)
    dAqk[:, P:] = 0
    dL[:, P:] = 0

    def run(inp):
        q, k, g_tok, beta = inp
        return bwd_intra_kda(q, k, chunk_cumsum_log2(g_tok, None), beta, dAqk, dL, scale=SCALE)

    a, b = run(x), run(pert)
    for name, u, w in zip(("dq", "dk", "dg", "dbeta"), a, b):
        assert torch.equal(u[:, :P], w[:, :P]), f"{name} leaks future tokens (P={P})"


def test_deterministic():
    q, k, g_tok, beta = make(1, 704, 2, 4)
    cu = cu_t(VARLEN_CU)
    g_c = chunk_cumsum_log2(g_tok, cu)
    dAqk, dL = random_dA(1, 704, 4, cu)
    a = bwd_intra_kda(q, k, g_c, beta, dAqk, dL, cu_seqlens=cu)
    for _ in range(5):
        b = bwd_intra_kda(q, k, g_c, beta, dAqk, dL, cu_seqlens=cu)
        assert all(torch.equal(u, w) for u, w in zip(a, b))


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


def _packed_cu(total=16384, row=4096, mean_doc=1024, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cu = [0]
    for r in range(total // row):
        pos = 0
        while pos < row:
            pos += max(1, int(torch.empty(1).exponential_(1.0, generator=gen).item() * mean_doc))
            if pos < row:
                cu.append(r * row + pos)
        cu.append((r + 1) * row)
    return cu


def test_timing():
    """Informational: [1, 16384, HK=8, HV=16, 128], packed docs. FLA's chunk_kda_bwd_intra is fed the same dAqk
    (and dL as its dAkk); it computes the same intra-chunk gradients but with its own conventions."""
    q, k, g_tok, beta = make(1, 16384, 8, 16)
    cu_list = _packed_cu()
    cu = cu_t(cu_list)
    g_c = chunk_cumsum_log2(g_tok, cu)
    dAqk, dL = random_dA(1, 16384, 16, cu)
    print(f"bwd_intra_kda [{len(cu_list) - 1} docs]: {_time(lambda: bwd_intra_kda(q, k, g_c, beta, dAqk, dL, cu_seqlens=cu)):.0f} us")
    try:
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    except ImportError:
        return
    z = lambda *s: torch.zeros(*s, device=DEVICE, dtype=torch.float32)  # noqa: E731
    dq0, dk0, dg0 = z(1, 16384, 16, 128), z(1, 16384, 16, 128), z(1, 16384, 16, 128)
    db0 = z(1, 16384, 16)
    for safe in (False, True):
        try:
            us = _time(lambda: chunk_kda_bwd_intra(q=q, k=k, g=g_c, beta=beta, dAqk=dAqk.float(), dAkk=dL.float(), dq=dq0, dk=dk0,
                                                   db=db0, dg=dg0, cu_seqlens=cu, safe_gate=safe))
            print(f"FLA chunk_kda_bwd_intra (safe_gate={safe}): {us:.0f} us")
        except Exception as e:  # signature / dtype differences between FLA versions
            print(f"FLA chunk_kda_bwd_intra (safe_gate={safe}) not comparable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# weighted-sum test: what the A_log / dt_bias gradients actually see
# ---------------------------------------------------------------------------

def gate_tok(mode, B, T, HV, seed=0):
    """Natural-log per-token gate like tests/kda_common.make_inputs + chunk._log_decay (gdn / kimi / eda)."""
    from mogan_kda.ops.kimi_delta_rule.chunk import _log_decay
    torch.manual_seed(seed)
    x = torch.randn(B, T, HV, 128, device=DEVICE)
    a_hi = 0.8 if mode == "gdn" else 4.0
    A_log = torch.empty(HV, device=DEVICE).uniform_(min(0.2, a_hi / 4), a_hi).log()
    dt_bias = torch.randn(HV, 128, device=DEVICE) * 0.5 - 1.0
    return _log_decay(x, mode, A_log, dt_bias, None if mode == "gdn" else -5.0)


def chunk_cumsum_nat(g_tok, cu):
    """C = chunk-local FORWARD cumsum of the natural-log gate (the weights of dg in the A_log / dt_bias gradients)."""
    C = torch.zeros_like(g_tok, dtype=F64)
    _, chunks, _ = TR._chunk_list(g_tok.shape[0], g_tok.shape[1], cu)
    for ch in chunks:
        sl = slice(ch["s"], ch["s"] + ch["L"])
        C[ch["b"], sl] = torch.cumsum(g_tok[ch["b"], sl].double(), 0)
    return C


def weighted_rel(dg_kernel, dg_ref, C):
    """Per head W = sum_{t, c} dg'_tc C_tc; returns ||W_kernel - W_ref|| / ||W_ref|| over the heads."""
    wk = (dg_kernel.double() * C).sum(dim=(0, 1, 3))
    wr = (dg_ref.double() * C).sum(dim=(0, 1, 3))
    return ((wk - wr).norm() / wr.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
@pytest.mark.parametrize("gate", ["gdn", "kimi", "eda"])
def test_weighted_sum_of_dg(B, T, HK, HV, cu_list, gate):
    """W(dg') vs fp64 bwd_intra_torch on identical inputs (dAqk / dL / partials from the real chain). Must be < 1e-2:
    the A_log / dt_bias gradients are weighted sums of dg' over all tokens and must not see the rounding of bf16 operands."""
    q, k, _, beta = make(B, T, HK, HV)
    g_tok = gate_tok(gate, B, T, HV)
    cu = cu_t(cu_list)
    g_c = chunk_cumsum_log2(g_tok, cu)
    dAqk, dL, *parts = real_dA(q, k, g_tok, beta, cu)
    ref = oracle(q, k, g_c, beta, dAqk, dL, parts, cu)
    C = chunk_cumsum_nat(g_tok, cu)
    n = T if cu_list is None else cu_list[-1]
    out = bwd_intra_kda(q, k, g_c, beta, dAqk, dL, *parts, scale=SCALE, cu_seqlens=cu)
    res = weighted_rel(out[2][:, :n], ref["dg"][:, :n], C[:, :n])
    print(f"[{gate}] W(dg') rel err: {res:.3e}")
    assert res < 1e-2, res
