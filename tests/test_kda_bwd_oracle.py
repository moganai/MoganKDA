"""CPU tests of the chunked torch forward / backward oracle (``mogan_kda/ops/kimi_delta_rule/chunk/torch_ref.py``).

The module is loaded by path (importing the ``chunk`` package would import tilelang). Everything runs in fp64 on tiny
shapes (K = V = 8): forward vs ``ref_kda.ref_chunk_delta_rule``, backward vs torch.autograd of that reference
(dq, dk, dv, dg, dbeta, dh0; dense / varlen with length-1 and unaligned documents / GVA / h0 + dht / state_v_first),
the anchored factorisation, the structural diagonal property and bitwise prefix invariance.
"""
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from ref_kda import ref_chunk_delta_rule

F64 = torch.float64
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "mogan_kda", "ops", "kimi_delta_rule", "chunk", "torch_ref.py")
_spec = importlib.util.spec_from_file_location("kda_torch_ref", _PATH)
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)

K = V = 8
FWD_TOL, BWD_TOL = 1e-10, 1e-9

# (B, T, HK, HV, cu_seqlens)
CASES = [
    pytest.param(1, 1, 1, 1, None, id="T1"),
    pytest.param(2, 17, 1, 2, None, id="T17-gva"),
    pytest.param(1, 64, 2, 2, None, id="T64"),
    pytest.param(2, 65, 1, 2, None, id="T65-gva"),
    pytest.param(1, 130, 1, 2, None, id="T130-gva"),
    pytest.param(1, 130, 1, 2, [0, 1, 18, 83, 130], id="varlen-1-17-65-47-gva"),
    pytest.param(1, 5, 1, 1, [0, 1, 2, 3, 4, 5], id="varlen-5x1"),
]


def make(B, T, HK, HV, seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, HK, K, dtype=F64), dim=-1)
    k = F.normalize(torch.randn(B, T, HK, K, dtype=F64), dim=-1)
    v = torch.randn(B, T, HV, V, dtype=F64)
    g = -5 * torch.sigmoid(torch.randn(B, T, HV, K, dtype=F64))  # per-token natural-log decay in [-5, 0)
    beta = 2 * torch.sigmoid(torch.randn(B, T, HV, dtype=F64))
    return q, k, v, g, beta


def cu_of(cu_list):
    return None if cu_list is None else torch.tensor(cu_list, dtype=torch.int32)


def states(cu_list, B, HV, vfirst, seed=3):
    torch.manual_seed(seed)
    n = B if cu_list is None else len(cu_list) - 1
    h0 = torch.randn(n, HV, K, V, dtype=F64) * 0.2
    dht = torch.randn(n, HV, K, V, dtype=F64) * 0.2
    if vfirst:
        h0, dht = h0.transpose(-1, -2).contiguous(), dht.transpose(-1, -2).contiguous()
    return h0, dht


def ref_grads(q, k, v, g, beta, do, h0, dht, cu, vfirst, scale):
    leaves = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
    h0_ = None if h0 is None else h0.detach().clone().requires_grad_()
    o, ht = ref_chunk_delta_rule(*leaves, scale=scale, initial_state=h0_, output_final_state=dht is not None,
                                 cu_seqlens=cu, state_v_first=vfirst)
    loss = (o * do).sum() + (0 if dht is None else (ht * dht).sum())
    ins = leaves + ([h0_] if h0_ is not None else [])
    return o, ht, torch.autograd.grad(loss, ins)


def mx(a, b):
    return (a - b).abs().max().item()


def close(a, b, tol):
    return mx(a, b) <= tol * (1.0 + b.abs().max().item())


@pytest.mark.parametrize("B, T, HK, HV, cu_list", CASES)
@pytest.mark.parametrize("use_state", [False, True], ids=["nostate", "h0dht"])
@pytest.mark.parametrize("vfirst", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("anchored", [False, True], ids=["direct", "anchored"])
def test_forward_and_backward_vs_reference_autograd(B, T, HK, HV, cu_list, use_state, vfirst, anchored):
    q, k, v, g, beta = make(B, T, HK, HV)
    cu = cu_of(cu_list)
    h0, dht = states(cu_list, B, HV, vfirst) if use_state else (None, None)
    scale = 0.37
    torch.manual_seed(11)
    do = torch.randn(B, T, HV, V, dtype=F64)

    o_ref, ht_ref, grads = ref_grads(q, k, v, g, beta, do, h0, dht, cu, vfirst, scale)
    fwd = TR.chunk_fwd_torch(q, k, v, g, beta, scale, h0, cu, vfirst, anchored=anchored)
    assert close(fwd["o"], o_ref, FWD_TOL), f"o {mx(fwd['o'], o_ref):.2e}"
    if ht_ref is not None:  # the reference returns the final state only when asked (dht given)
        assert close(fwd["ht"], ht_ref, FWD_TOL), f"ht {mx(fwd['ht'], ht_ref):.2e}"

    r = TR.chunk_bwd_torch(q, k, v, beta, fwd["g_c"], fwd["A"], fwd["Aqk"], fwd["h"], do, dht, scale, cu, vfirst,
                           anchored=anchored)
    names = ("dq", "dk", "dv", "dg", "dbeta") + (("dh0",) if h0 is not None else ())
    for name, ref in zip(names, grads):
        assert close(r[name], ref, BWD_TOL), f"{name}: {mx(r[name], ref):.2e}"


def test_direct_and_anchored_agree():
    q, k, v, g, beta = make(1, 130, 1, 2, seed=5)
    do = torch.randn(1, 130, 2, V, dtype=F64)
    out = {}
    for anchored in (False, True):
        f = TR.chunk_fwd_torch(q, k, v, g, beta, anchored=anchored)
        out[anchored] = TR.chunk_bwd_torch(q, k, v, beta, f["g_c"], f["A"], f["Aqk"], f["h"], do, anchored=anchored)
    for name in ("dq", "dk", "dv", "dg", "dbeta"):
        assert close(out[True][name], out[False][name], BWD_TOL), name


# ---------------------------------------------------------------------------
# structural diagonal property
# ---------------------------------------------------------------------------

def _diag_only(B, T, HV, cu_list, seed=1):
    torch.manual_seed(seed)
    dAqk = torch.zeros(B, T, HV, 64, dtype=F64)
    _, chunks, _ = TR._chunk_list(B, T, cu_of(cu_list))
    for ch in chunks:
        for i in range(ch["L"]):
            dAqk[ch["b"], ch["s"] + i, :, i] = torch.randn(HV, dtype=F64)
    return dAqk


@pytest.mark.parametrize("anchored", [False, True], ids=["direct", "anchored"])
@pytest.mark.parametrize("B, T, HK, HV, cu_list", [pytest.param(1, 5, 1, 1, [0, 1, 2, 3, 4, 5], id="1-token-chunks"),
                                                    pytest.param(1, 130, 1, 2, None, id="T130"),
                                                    pytest.param(1, 130, 1, 2, [0, 1, 18, 83, 130], id="varlen")])
def test_diagonal_gives_exactly_zero_dg(B, T, HK, HV, cu_list, anchored):
    q, k, _, g, beta = make(B, T, HK, HV)
    cu = cu_of(cu_list)
    g_c = torch.zeros(B, T, HV, K, dtype=F64)
    _, chunks, _ = TR._chunk_list(B, T, cu)
    for ch in chunks:
        sl = slice(ch["s"], ch["s"] + ch["L"])
        g_c[ch["b"], sl] = torch.cumsum(g[ch["b"], sl], 0) * TR.RCP_LN2
    dAqk = _diag_only(B, T, HV, cu_list)
    zero = torch.zeros(B, T, HV, K, dtype=F64)
    r = TR.bwd_intra_torch(q, k, g_c, beta, dAqk, torch.zeros_like(dAqk), zero, zero, zero, 0.5, cu, anchored=anchored)
    assert torch.equal(r["dg"], torch.zeros_like(r["dg"])), "diagonal dAqk leaked into dg"
    diag = torch.stack([dAqk[:, t, :, t % 64] for t in range(T)], 1) if cu_list is None else None
    kx = k.repeat_interleave(HV // HK, 2)
    if cu_list is None:  # dq_i = scale dAqk_ii k_i exactly (gate free)
        assert torch.equal(r["dq"], 0.5 * diag[..., None] * kx)


def test_diagonal_of_dAqk_never_changes_dg():
    """dg computed with a full dAqk equals dg computed with its diagonal removed, bitwise."""
    B, T, HK, HV = 1, 130, 1, 2
    q, k, _, g, beta = make(B, T, HK, HV, seed=2)
    torch.manual_seed(4)
    g_c = torch.zeros(B, T, HV, K, dtype=F64)
    for c in range(0, T, 64):
        g_c[0, c:c + 64] = torch.cumsum(g[0, c:c + 64], 0) * TR.RCP_LN2
    dAqk = torch.tril(torch.randn(HV, 64, 64, dtype=F64)).permute(1, 0, 2)[None].repeat(1, 3, 1, 1)[:, :T]
    dL = torch.tril(torch.randn(HV, 64, 64, dtype=F64), -1).permute(1, 0, 2)[None].repeat(1, 3, 1, 1)[:, :T]
    nodiag = dAqk.clone()
    for t in range(T):
        nodiag[:, t, :, t % 64] = 0
    zero = torch.zeros(B, T, HV, K, dtype=F64)
    for anchored in (False, True):
        a = TR.bwd_intra_torch(q, k, g_c, beta, dAqk, dL, zero, zero, zero, anchored=anchored)
        b = TR.bwd_intra_torch(q, k, g_c, beta, nodiag, dL, zero, zero, zero, anchored=anchored)
        assert torch.equal(a["dg"], b["dg"]) and torch.equal(a["dbeta_g"], b["dbeta_g"])


# ---------------------------------------------------------------------------
# bitwise prefix invariance of the oracle's gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchored", [False, True], ids=["direct", "anchored"])
@pytest.mark.parametrize("P", [1, 17, 63, 65])
def test_prefix_invariance_bitwise(P, anchored):
    B, T, HK, HV = 1, 130, 1, 2
    x, y = make(B, T, HK, HV, seed=0), make(B, T, HK, HV, seed=1)
    pert = tuple(torch.cat([a[:, :P], b[:, P:]], 1) for a, b in zip(x, y))
    torch.manual_seed(9)
    do = torch.randn(B, T, HV, V, dtype=F64)
    do[:, P:] = 0

    def run(inp):
        q, k, v, g, beta = inp
        f = TR.chunk_fwd_torch(q, k, v, g, beta, anchored=anchored)
        return TR.chunk_bwd_torch(q, k, v, beta, f["g_c"], f["A"], f["Aqk"], f["h"], do, anchored=anchored)

    a, b = run(x), run(pert)
    for name in ("dq", "dk", "dv", "dg", "dbeta"):
        assert torch.equal(a[name][:, :P], b[name][:, :P]), f"{name} leaks future tokens (P={P})"
