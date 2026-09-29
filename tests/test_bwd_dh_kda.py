"""Tests for K0 ``scaled_operands_kda``, K1 ``bwd_w_kda`` and K2 ``bwd_dh_kda`` (the reverse state chain).

Oracles: ``torch_ref.scaled_operands_torch`` / ``bwd_w_torch`` / ``bwd_dh_torch`` in fp64, fed with EXACTLY the tensors the
kernels consume (forward outputs A / Aqk / h of ``chunk_kda_fwd`` and the K0 / K1 outputs, converted to fp64).
Metric: max|d| / max|ref| per output < 5e-3 (values printed). ``dgl`` is not produced by K2 (K3b computes it).
"""
import pytest
import torch
import torch.nn.functional as F

from mogan_kda.ops.kimi_delta_rule.chunk import chunk_kda_fwd
from mogan_kda.ops.kimi_delta_rule.chunk import torch_ref as TR
from mogan_kda.ops.kimi_delta_rule.chunk.hopper import bwd_dh_kda, bwd_w_kda, scaled_operands_kda
from mogan_kda.utils import prepare_chunk_offsets

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]

DEVICE = "cuda"
TOL = 5e-3
SCALE = 128 ** -0.5
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs 1, 97, 256, 150, 200

CONFIGS = [
    pytest.param(2, 300, 2, 4, None, id="dense-B2-T300-Hk2Hv4"),
    pytest.param(1, 704, 2, 4, VARLEN_CU, id="varlen-Hk2Hv4"),
]


def make(B, T, HK, HV, seed=0):
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEVICE, dtype=torch.float32)  # noqa: E731
    q = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    k = F.normalize(r(B, T, HK, 128), dim=-1).bfloat16()
    v = r(B, T, HV, 128).bfloat16()
    g = -5 * torch.sigmoid(r(B, T, HV, 128))  # per-token natural-log decay in [-5, 0)
    beta = 2 * torch.sigmoid(r(B, T, HV))
    return q, k, v, g, beta


def cu_t(cu_list):
    return None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)


def n_seq(B, cu_list):
    return B if cu_list is None else len(cu_list) - 1


def make_states(B, HV, cu_list, seed=3):
    torch.manual_seed(seed)
    n = n_seq(B, cu_list)
    h0 = 0.1 * torch.randn(n, HV, 128, 128, device=DEVICE)
    dht = 0.1 * torch.randn(n, HV, 128, 128, device=DEVICE)
    return h0, dht


def make_do(B, T, HV, seed=9):
    torch.manual_seed(seed)
    return torch.randn(B, T, HV, 128, device=DEVICE).bfloat16()


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def pipeline(q, k, v, g, beta, cu, h0=None, vfirst=False):
    """Forward + K0 + K1: everything K2 (and its oracle) needs."""
    _, _, g_c, A, Aqk, h = chunk_kda_fwd(q, k, v, g, beta, scale=SCALE, initial_state=h0, cu_seqlens=cu,
                                         state_v_first=vfirst, output_h=True)
    qt, kt, kb, gl = scaled_operands_kda(q, k, g_c, cu)
    nw = bwd_w_kda(A, beta, kt, cu, negate=True)
    return dict(g_c=g_c, A=A, Aqk=Aqk, h=h, qt=qt, kt=kt, kb=kb, gl=gl, nw=nw)


def to_k(x, vfirst):
    return None if x is None else (x.transpose(-1, -2).contiguous() if vfirst else x)


# ---------------------------------------------------------------------------
# K0 / K1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
def test_k0_scaled_operands(B, T, HK, HV, cu_list):
    q, k, v, g, beta = make(B, T, HK, HV)
    cu = cu_t(cu_list)
    p = pipeline(q, k, v, g, beta, cu)
    ref = TR.scaled_operands_torch(q, k, p["g_c"], cu)
    n = T if cu_list is None else cu_list[-1]
    errs = {"qt": rel(p["qt"][:, :n], ref[0][:, :n]), "kt": rel(p["kt"][:, :n], ref[1][:, :n]),
            "kb": rel(p["kb"][:, :n], ref[2][:, :n]), "gl": rel(p["gl"], ref[3])}
    print(errs)
    assert all(e < TOL for e in errs.values()) and errs["gl"] < 1e-5


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
def test_k1_w(B, T, HK, HV, cu_list):
    q, k, v, g, beta = make(B, T, HK, HV)
    cu = cu_t(cu_list)
    p = pipeline(q, k, v, g, beta, cu)
    w = bwd_w_kda(p["A"], beta, p["kt"], cu)
    ref = TR.bwd_w_torch(p["A"], beta, p["kt"], cu)
    n = T if cu_list is None else cu_list[-1]
    e = rel(w[:, :n], ref[:, :n])
    print(f"w rel err {e:.2e}")
    assert e < TOL
    # negate=True is -w up to fast-math flush-to-zero: the `* -1.0` epilogue flushes fp32 subnormals (|x| < 2**-126)
    # to zero (w itself is deterministic), while the non-negated fp32 subnormal accumulator rounds UP to the bf16
    # min-normal 2**-126 on conversion. Compare after flushing |x| <= 2**-126 on both sides (bitwise equal otherwise).
    ftz = lambda x: torch.where(x.float().abs() <= 2.0 ** -126, torch.zeros_like(x), x)  # noqa: E731
    assert torch.equal(ftz(p["nw"]), ftz(-w)), "negate=True must return -w (up to subnormal flush)"


# ---------------------------------------------------------------------------
# K2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
@pytest.mark.parametrize("use_state", [False, True], ids=["nostate", "h0dht"])
@pytest.mark.parametrize("vfirst", [False, True], ids=["kv", "vk"])
def test_k2_vs_oracle(B, T, HK, HV, cu_list, use_state, vfirst):
    q, k, v, g, beta = make(B, T, HK, HV)
    cu = cu_t(cu_list)
    h0, dht = make_states(B, HV, cu_list) if use_state else (None, None)
    p = pipeline(q, k, v, g, beta, cu, to_k(h0, vfirst), vfirst)
    do = make_do(B, T, HV)
    dh, dvd, dh0 = bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], to_k(dht, vfirst), SCALE, cu,
                              vfirst, output_dh0=use_state)
    ref = TR.bwd_dh_torch(do, p["Aqk"], -p["nw"].double(), p["qt"], p["kb"], p["gl"], p["h"], to_k(dht, vfirst),
                          SCALE, cu, vfirst)
    n = T if cu_list is None else cu_list[-1]
    errs = {"dh": rel(dh, ref["dh"]), "dvd": rel(dvd[:, :n], ref["dVd"][:, :n])}
    if use_state:
        errs["dh0"] = rel(dh0, ref["dh0"])
    print(errs)
    assert all(e < TOL for e in errs.values()), errs


@pytest.mark.parametrize("P", [1, 15, 16, 17, 63, 65, 193])
def test_k2_prefix_invariance_bitwise(P):
    """dO = 0 (and dht = 0) from row P on: dh and dVd of the whole tensor are bitwise independent of the future tokens
    (all chunks from the one containing P on carry exact zeros, and dVd rows >= P are exactly 0)."""
    B, T, HK, HV = 1, 256, 2, 4
    x, y = make(B, T, HK, HV, seed=0), make(B, T, HK, HV, seed=1)
    pert = tuple(torch.cat([a[:, :P], b[:, P:]], 1) for a, b in zip(x, y))
    do = make_do(B, T, HV)
    do[:, P:] = 0

    def run(inp):
        p = pipeline(*inp, None)
        return bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], None, SCALE, None)

    dh0_, dvd0 = run(x)[:2]
    dh1_, dvd1 = run(pert)[:2]
    assert dvd0[:, :P].abs().max() > 0, "test is vacuous"
    assert torch.equal(dh0_, dh1_), f"dh leaks future tokens (P={P})"
    assert torch.equal(dvd0, dvd1), f"dVd leaks future tokens (P={P})"


@pytest.mark.parametrize("doc", [0, 2, 4])
def test_k2_document_isolation_bitwise(doc):
    B, T, HK, HV = 1, 704, 2, 4
    x, y = make(B, T, HK, HV, seed=0), make(B, T, HK, HV, seed=1)
    s, e = VARLEN_CU[doc], VARLEN_CU[doc + 1]

    def splice(a, b):
        out = a.clone()
        out[:, s:e] = b[:, s:e]
        return out

    pert = tuple(splice(a, b) for a, b in zip(x, y))
    cu = cu_t(VARLEN_CU)
    h0, dht = make_states(B, HV, VARLEN_CU)
    do0 = make_do(B, T, HV)
    do1 = splice(do0, make_do(B, T, HV, seed=10))

    def run(inp, do):
        p = pipeline(*inp, cu, h0)
        return bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], dht, SCALE, cu, output_dh0=True)

    dh_a, dvd_a, dh0_a = run(x, do0)
    dh_b, dvd_b, dh0_b = run(pert, do1)
    keep = torch.ones(T, dtype=torch.bool, device=DEVICE)
    keep[s:e] = False
    docs = [d for d in range(len(VARLEN_CU) - 1) if d != doc]
    off, _ = prepare_chunk_offsets(cu, 64)
    chunk_keep = torch.ones(int(off[-1]), dtype=torch.bool, device=DEVICE)
    chunk_keep[int(off[doc]):int(off[doc + 1])] = False
    assert torch.equal(dvd_a[:, keep], dvd_b[:, keep]) and torch.equal(dh_a[:, chunk_keep], dh_b[:, chunk_keep])
    assert torch.equal(dh0_a[docs], dh0_b[docs])
    # non-vacuity: something of the perturbed doc must change (for a 1-token doc dh equals the unchanged dht, but dVd changes)
    changed = (not torch.equal(dh_a[:, ~chunk_keep], dh_b[:, ~chunk_keep])
               or not torch.equal(dvd_a[:, ~keep], dvd_b[:, ~keep])
               or not torch.equal(dh0_a[doc], dh0_b[doc]))
    assert changed, "perturbation had no effect (vacuous)"


def test_k2_deterministic():
    B, T, HK, HV = 1, 704, 2, 4
    q, k, v, g, beta = make(B, T, HK, HV)
    cu = cu_t(VARLEN_CU)
    h0, dht = make_states(B, HV, VARLEN_CU)
    p = pipeline(q, k, v, g, beta, cu, h0)
    do = make_do(B, T, HV)
    a = bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], dht, SCALE, cu, output_dh0=True)
    for _ in range(5):
        b = bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], dht, SCALE, cu, output_dh0=True)
        assert all(torch.equal(u, w) for u, w in zip(a, b))


# ---------------------------------------------------------------------------
# timing
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
    """Informational, [1, 16384, HK=8, HV=16, 128], packed docs."""
    q, k, v, g, beta = make(1, 16384, 8, 16)
    cu_list = _packed_cu()
    cu = cu_t(cu_list)
    p = pipeline(q, k, v, g, beta, cu)
    do = make_do(1, 16384, 16)
    t0 = _time(lambda: scaled_operands_kda(q, k, p["g_c"], cu))
    t1 = _time(lambda: bwd_w_kda(p["A"], beta, p["kt"], cu, negate=True))
    t2 = _time(lambda: bwd_dh_kda(do, p["Aqk"], p["nw"], p["qt"], p["kb"], p["gl"], None, SCALE, cu))
    print(f"[{len(cu_list) - 1} docs] K0 {t0:.0f} us | K1 {t1:.0f} us | K2 {t2:.0f} us | total {t0 + t1 + t2:.0f} us")
