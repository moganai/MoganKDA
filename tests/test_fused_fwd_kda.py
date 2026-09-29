"""Tests for ``fused_fwd_kda`` and the forward chain ``chunk_kda_fwd`` (cumsum -> kkt_solve_kda -> fused_fwd_kda).

(1) fused kernel alone vs an fp64 chunk reference fed with the SAME A / Aqk / qt / kt / kb / gl (isolates the fused
    kernel), incl. per-chunk states h (store_h) and final state, dense + varlen + GVA + h0 + state_v_first;
(2) end to end vs ``ref_kda.ref_chunk_delta_rule`` with the training metric (fwd max < 2e-2, mean < 5e-3);
(3) bitwise prefix invariance, (4) document isolation, (5) determinism, (6) emit_scaled=False unchanged, (7) timing.
"""
import pytest
import torch
import torch.nn.functional as F

from mogan_kda.ops.kimi_delta_rule.chunk import chunk_kda_fwd
from mogan_kda.ops.kimi_delta_rule.chunk.hopper import fused_fwd_kda, kkt_solve_kda
from mogan_kda.ops.utils import chunk_local_cumsum
from kda_common import THRESH, rel_err
from ref_kda import ref_chunk_delta_rule

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]

DEVICE = "cuda"
CHUNK = 64
RCP_LN2 = 1.4426950408889634
SCALE = 128 ** -0.5
VARLEN_CU = [0, 1, 98, 354, 504, 704]  # docs 1, 97, 256, 150, 200


def make_inputs(B, T, HK, HV, seed=0):
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


def make_h0(cu_list, B, HV, seed=7):
    torch.manual_seed(seed)
    n = B if cu_list is None else len(cu_list) - 1
    return torch.randn(n, HV, 128, 128, device=DEVICE, dtype=torch.float32) * 0.1


def seqs(B, T, cu_list):
    return [(b, 0, T) for b in range(B)] if cu_list is None else [(0, cu_list[i], cu_list[i + 1]) for i in range(len(cu_list) - 1)]


def ref_fused(qt, kt, kb, v, A, Aqk, gl, beta, scale, h0, cu_list):
    """fp64 reference of the fused kernel on the given operands. Returns o [B,T,H,V], ht [N,H,K,V] and h with the
    layout of the kernel output ([B, NC, H, K, V]; varlen [1, total_chunks])."""
    B, T, H, _ = qt.shape
    f = lambda x: x.double()  # noqa: E731
    o = torch.zeros(B, T, H, 128, dtype=torch.float64, device=qt.device)
    hs, ht, chunk_off = {}, [], 0
    for n, (b, s, e) in enumerate(seqs(B, T, cu_list)):
        S = f(h0[n]) if h0 is not None else torch.zeros(H, 128, 128, dtype=torch.float64, device=qt.device)
        for ci, c in enumerate(range(s, e, CHUNK)):
            L = min(CHUNK, e - c)
            sl = slice(c, c + L)
            hs[(b, chunk_off + ci)] = S.clone()
            W = f(beta[b, sl])[:, :, None] * (f(v[b, sl]) - torch.einsum("lhk,hkv->lhv", f(kt[b, sl]), S))
            Vd = torch.einsum("ihj,jhv->ihv", f(A[b, sl, :, :L]), W)
            o[b, sl] = scale * torch.einsum("lhk,hkv->lhv", f(qt[b, sl]), S) + \
                torch.einsum("ihj,jhv->ihv", f(Aqk[b, sl, :, :L]), Vd)
            gi = ci if cu_list is None else chunk_off + ci  # gl chunk id, laid out like h
            S = S * f(gl[b, gi])[:, :, None] + torch.einsum("lhk,lhv->hkv", f(kb[b, sl]), Vd)
        chunk_off += 0 if cu_list is None else (e - s + CHUNK - 1) // CHUNK
        ht.append(S)
    nc = (T + CHUNK - 1) // CHUNK
    if cu_list is None:
        h = torch.stack([torch.stack([hs[(b, c)] for c in range(nc)]) for b in range(B)])
    else:
        h = torch.stack([hs[(0, c)] for c in range(chunk_off)])[None]
    return o, torch.stack(ht), h


def pipeline_operands(q, k, g, beta, cu, scale=SCALE):
    g_c = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2)
    return g_c, *kkt_solve_kda(q, k, g_c, beta, scale=scale, cu_seqlens=cu, emit_scaled=True)


CONFIGS = [
    pytest.param(2, 300, 2, 4, None, id="dense-B2-T300-Hk2Hv4"),
    pytest.param(1, 704, 2, 4, VARLEN_CU, id="varlen-Hk2Hv4"),
]


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True], ids=["noh0", "h0"])
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_fused_vs_fp64_chunk_reference(B, T, HK, HV, cu_list, use_h0, state_v_first):
    q, k, v, g, beta = make_inputs(B, T, HK, HV)
    cu = cu_t(cu_list)
    _, A, Aqk, qt, kt, kb, gl = pipeline_operands(q, k, g, beta, cu)
    h0 = make_h0(cu_list, B, HV) if use_h0 else None
    o_ref, ht_ref, h_ref = ref_fused(qt, kt, kb, v, A, Aqk, gl, beta, SCALE, h0, cu_list)
    h0_k = h0.transpose(-1, -2).contiguous() if (use_h0 and state_v_first) else h0
    o, h, ht = fused_fwd_kda(qt, kt, kb, v, A, Aqk, gl, beta, scale=SCALE, initial_state=h0_k,
                             output_final_state=True, output_h=True, cu_seqlens=cu, state_v_first=state_v_first)
    if state_v_first:
        ht, h = ht.transpose(-1, -2), h.transpose(-1, -2)
    n = T if cu_list is None else cu_list[-1]
    errs = {"o": rel_err(o[:, :n], o_ref[:, :n]), "ht": rel_err(ht, ht_ref), "h": rel_err(h.float(), h_ref)}
    print({k_: f"max {a:.2e} mean {b:.2e}" for k_, (a, b) in errs.items()})
    for name, (mx, mean) in errs.items():
        assert mx < THRESH["fwd_max"] and mean < THRESH["fwd_mean"], (name, mx, mean)
    if cu_list is not None and T > cu_list[-1]:
        assert (o[:, cu_list[-1]:] == 0).all()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B, T, HK, HV, cu_list", CONFIGS + [pytest.param(1, 704 + 37, 2, 4, VARLEN_CU, id="varlen-padded-tail")])
@pytest.mark.parametrize("use_h0", [False, True], ids=["noh0", "h0"])
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_end_to_end_vs_ref(B, T, HK, HV, cu_list, use_h0, state_v_first):
    q, k, v, g, beta = make_inputs(B, T, HK, HV)
    cu = cu_t(cu_list)
    h0 = make_h0(cu_list, B, HV) if use_h0 else None
    h0_k = h0.transpose(-1, -2).contiguous() if (use_h0 and state_v_first) else h0
    o, ht, *_ = chunk_kda_fwd(q, k, v, g, beta, initial_state=h0_k, output_final_state=True, cu_seqlens=cu,
                              state_v_first=state_v_first)
    if state_v_first:
        ht = ht.transpose(-1, -2)
    n = T if cu_list is None else cu_list[-1]
    o_ref, ht_ref = ref_chunk_delta_rule(q[:, :n], k[:, :n], v[:, :n], g[:, :n], beta[:, :n], initial_state=h0,
                                         output_final_state=True, cu_seqlens=None if cu_list is None else cu[:].cpu())
    fm, fo = rel_err(o[:, :n], o_ref)
    sm, so = rel_err(ht, ht_ref)
    print(f"o max {fm:.2e} mean {fo:.2e} | state max {sm:.2e} mean {so:.2e}")
    assert fm < THRESH["fwd_max"] and fo < THRESH["fwd_mean"]
    assert sm < THRESH["fwd_max"] and so < THRESH["fwd_mean"]


# ---------------------------------------------------------------------------
# bitwise structure tests
# ---------------------------------------------------------------------------

def splice(a, b, s, e=None):
    out = a.clone()
    out[:, s:e] = b[:, s:e]
    return out


@pytest.mark.parametrize("P", [1, 15, 16, 17, 63, 65, 193])
def test_prefix_invariance_bitwise(P):
    """Dense: perturbing tokens >= P leaves o[:, :P] bitwise unchanged. Varlen [0, P, T): perturbing the second
    document leaves o of the first document and its final state bitwise unchanged."""
    B, T = 1, 256
    x, y = make_inputs(B, T, 2, 4, seed=0), make_inputs(B, T, 2, 4, seed=1)
    pert = tuple(splice(a, b, P) for a, b in zip(x, y))
    o0, *_ = chunk_kda_fwd(*x)
    o1, *_ = chunk_kda_fwd(*pert)
    assert torch.equal(o0[:, :P], o1[:, :P]), f"o leaks future tokens (P={P})"

    cu = cu_t([0, P, T])
    h0 = make_h0([0, P, T], B, 4)
    a = chunk_kda_fwd(*x, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    b = chunk_kda_fwd(*pert, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    assert torch.equal(a[0][:, :P], b[0][:, :P]) and torch.equal(a[1][0], b[1][0]), f"doc isolation broken (P={P})"


@pytest.mark.parametrize("doc", [0, 2, 4])
def test_varlen_document_isolation_bitwise(doc):
    B, T = 1, 704
    x, y = make_inputs(B, T, 2, 4, seed=0), make_inputs(B, T, 2, 4, seed=1)
    s, e = VARLEN_CU[doc], VARLEN_CU[doc + 1]
    pert = tuple(splice(a, b, s, e) for a, b in zip(x, y))
    cu = cu_t(VARLEN_CU)
    h0 = make_h0(VARLEN_CU, B, 4)
    o0, ht0, *_ = chunk_kda_fwd(*x, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    o1, ht1, *_ = chunk_kda_fwd(*pert, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    keep = torch.ones(T, dtype=torch.bool, device=DEVICE)
    keep[s:e] = False
    docs = [d for d in range(len(VARLEN_CU) - 1) if d != doc]
    assert torch.equal(o0[:, keep], o1[:, keep]) and torch.equal(ht0[docs], ht1[docs])
    assert not torch.equal(o0[:, ~keep], o1[:, ~keep]), "perturbation had no effect (test is vacuous)"


def test_deterministic():
    x = make_inputs(1, 704, 2, 4)
    cu = cu_t(VARLEN_CU)
    h0 = make_h0(VARLEN_CU, 1, 4)
    o0, ht0, *_ = chunk_kda_fwd(*x, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    for _ in range(5):
        o1, ht1, *_ = chunk_kda_fwd(*x, initial_state=h0, output_final_state=True, cu_seqlens=cu)
        assert torch.equal(o0, o1) and torch.equal(ht0, ht1)


def test_kkt_emit_scaled_false_unchanged():
    q, k, v, g, beta = make_inputs(1, 704, 2, 4)
    cu = cu_t(VARLEN_CU)
    g_c = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2)
    A0, Q0 = kkt_solve_kda(q, k, g_c, beta, cu_seqlens=cu)
    A1, Q1, *_ = kkt_solve_kda(q, k, g_c, beta, cu_seqlens=cu, emit_scaled=True)
    assert torch.equal(A0, A1) and torch.equal(Q0, Q1)


def test_emit_scaled_values():
    """qt / kt / kb / gl against direct torch formulas (dense, partial last chunk)."""
    q, k, v, g, beta = make_inputs(1, 130, 2, 4)
    g_c, A, Aqk, qt, kt, kb, gl = pipeline_operands(q, k, g, beta, None)
    G = torch.exp2(g_c.double())
    qx, kx = q.double().repeat_interleave(2, 2), k.double().repeat_interleave(2, 2)
    for c in range(0, 130, CHUNK):
        sl = slice(c, min(c + CHUNK, 130))
        gL = g_c[:, sl][:, -1:].double()  # [1, 1, H, K]
        refs = {"qt": (qt, qx[:, sl] * G[:, sl]), "kt": (kt, kx[:, sl] * G[:, sl]),
                "kb": (kb, kx[:, sl] * torch.exp2(gL - g_c[:, sl].double()))}
        for name, (got, ref) in refs.items():
            assert rel_err(got[:, sl].float(), ref)[0] < 1e-2, name
        assert rel_err(gl[0, c // CHUNK], torch.exp2(gL[0, 0]))[0] < 1e-6


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def _time(fn, iters=30):
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


def _training_like_cu(total=16384, row=4096, mean_doc=1024, seed=0):
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
    """Informational: [1, 16384, HK=8, HV=16, 128], packed training-like docs (and one single 16k doc)."""
    x = make_inputs(1, 16384, 8, 16)
    q, k, v, g, beta = x
    for name, cu_list in (("packed docs", _training_like_cu()), ("single 16k doc", None)):
        cu = cu_t(cu_list)
        g_c = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2)
        A, Aqk, qt, kt, kb, gl = kkt_solve_kda(q, k, g_c, beta, cu_seqlens=cu, emit_scaled=True)
        t_cs = _time(lambda: chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu, scale=RCP_LN2))
        t_kkt = _time(lambda: kkt_solve_kda(q, k, g_c, beta, cu_seqlens=cu, emit_scaled=True))
        t_fused = _time(lambda: fused_fwd_kda(qt, kt, kb, v, A, Aqk, gl, beta, cu_seqlens=cu, output_final_state=False))
        t_total = _time(lambda: chunk_kda_fwd(q, k, v, g, beta, cu_seqlens=cu))
        print(f"[{name}, {0 if cu_list is None else len(cu_list) - 1} docs] cumsum {t_cs:.0f} us | kkt(emit) {t_kkt:.0f} us | "
              f"fused_fwd_kda {t_fused:.0f} us | chain total {t_total:.0f} us")
        try:
            from fla.ops.kda import chunk_kda
        except ImportError:
            continue
        fla_us = _time(lambda: chunk_kda(q, k, v, g, beta, scale=SCALE, use_qk_l2norm_in_kernel=False,
                                         use_gate_in_kernel=False, cu_seqlens=cu))
        print(f"    FLA chunk_kda forward total: {fla_us:.0f} us")
