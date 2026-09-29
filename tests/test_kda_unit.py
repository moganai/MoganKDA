"""GPU correctness harness for the unified delta-rule API (KDA per-channel gate, GVA, varlen, options).

Implementation under test: env ``KDA_IMPL`` (comma list, default ``qla``): ``qla`` = the mogan_kda KDA entry point,
``fla`` = FLA ``chunk_kda`` (second reference). Everything is compared against the fp64 reference ``ref_kda.ref_chunk_delta_rule``
with the training-test metric (fwd max < 2e-2, fwd mean < 5e-3, grad max per tensor < 5e-2).

    pytest tests/test_kda_unit.py -s                       # reduced pairwise matrix, mogan_kda
    KDA_IMPL=qla,fla KDA_FULL=1 pytest tests/test_kda_unit.py
    KDA_REPORT=kda_report.json pytest tests/test_kda_unit.py -s   # all measured metrics as JSON
"""
import dataclasses
import functools
import os

import pytest
import torch

from kda_common import (
    THRESH, TOKEN_LEAVES, Case, ImplUnavailable, dims_for, doc_slice, epi_weighted_err, full_matrix, grad_errs, load_impl,
    make_inputs, make_run, record, reduced_matrix, rel_err, run_impl, run_ref, write_report,
)

IMPLS = [s for s in os.environ.get("KDA_IMPL", "qla").split(",") if s]
CASES = full_matrix() if os.environ.get("KDA_FULL") == "1" else reduced_matrix()

impl_param = pytest.mark.parametrize("impl_name", IMPLS)
case_param = pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in CASES])


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    write_report()


def get_impl(name):
    try:
        return load_impl(name)
    except ImplUnavailable as e:
        pytest.skip(str(e))


@functools.lru_cache(maxsize=64)
def _pair(impl_name, case):
    """(impl result, fp64 reference result) for one (impl, case); shared by test_fwd / test_bwd."""
    impl = get_impl(impl_name)
    inp = make_inputs(case)
    run = make_run(case)
    return run_impl(impl, inp, run, backward=impl.supports_bwd), run_ref(inp, run, backward=impl.supports_bwd)


def _check_fwd(tag, res, ref, extra=None):
    fm, fo = rel_err(res.o, ref.o)
    m = dict(test=tag, fwd_max=fm, fwd_mean=fo, notes=res.notes, **(extra or {}))
    if ref.o_pre is not None:  # epilogue output: pass/fail on the pre-norm scale, raw metric is info only
        m.update(fwd_max_raw=fm, fwd_mean_raw=fo)
        fm, fo = epi_weighted_err(res, ref)
        m.update(fwd_max=fm, fwd_mean=fo)
    ok = fm < THRESH["fwd_max"] and fo < THRESH["fwd_mean"]
    if ref.ht is not None:
        sm, so = rel_err(res.ht, ref.ht)
        m.update(state_max=sm, state_mean=so)
        ok = ok and sm < THRESH["fwd_max"] and so < THRESH["fwd_mean"]
    record(**m)
    assert ok, f"{tag}: {m}"


def _check_grads(tag, res, ref, extra=None):
    errs = grad_errs(res, ref)
    worst = max(errs.values())
    record(test=tag, grad_worst=worst, grads=errs, notes=res.notes, **(extra or {}))
    assert worst < THRESH["grad_max"], f"{tag}: worst {max(errs, key=errs.get)}={worst:.3e}, all {errs}"


# ---------------------------------------------------------------------------
# 1. forward / backward vs fp64 reference over the option matrix
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@impl_param
@case_param
def test_fwd(impl_name, case):
    res, ref = _pair(impl_name, case)
    _check_fwd(f"fwd[{impl_name}:{case.id}]", res, ref, dict(impl=impl_name))


@pytest.mark.gpu
@pytest.mark.needs_bwd
@impl_param
@case_param
def test_bwd(impl_name, case):
    if not get_impl(impl_name).supports_bwd:
        pytest.skip(f"{impl_name}: backward not implemented yet")
    res, ref = _pair(impl_name, case)
    _check_grads(f"bwd[{impl_name}:{case.id}]", res, ref, dict(impl=impl_name))


# ---------------------------------------------------------------------------
# 2. prefix invariance: perturbing future tokens must not change prefix outputs / prefix grads (bitwise)
# ---------------------------------------------------------------------------

PREFIX_GATES = ["precomputed", "kimi", "eda", "kimi_safe"]
PREFIX_POS = [1, 15, 17, 63, 77, 193]  # chunk-internal offsets 1 / 15 / 17 / 63 and unaligned prefixes


@pytest.mark.gpu
@impl_param
@pytest.mark.parametrize("gate", PREFIX_GATES)
@pytest.mark.parametrize("P", PREFIX_POS)
def test_prefix_invariance(impl_name, gate, P, request):
    impl = get_impl(impl_name)
    if impl_name == "fla" and gate == "kimi_safe":
        request.applymarker(pytest.mark.xfail(strict=False, reason="FLA #1240 (safe_gate + in-kernel gate leaks; P = 1 mod 16 observed on FLA 0.6.0)"))
    case = Case(gate="kimi" if gate == "kimi_safe" else gate, beta="s2", gva=True, l2=True)
    dims = dims_for(case, B=1, T=256)
    inp, other = make_inputs(case, dims, seed=0), make_inputs(case, dims, seed=1)
    inp.do[:, P:] = 0  # loss only on the prefix
    pert = dataclasses.replace(inp, leaves={
        n: (torch.cat([t[:, :P], other.leaves[n][:, P:]], 1) if n in TOKEN_LEAVES else t)
        for n, t in inp.leaves.items()})
    run = make_run(case, safe_gate=gate == "kimi_safe")
    bwd = impl.supports_bwd  # forward-only impls: output invariance only
    base, base2, pt = (run_impl(impl, i, run, backward=bwd) for i in (inp, inp, pert))
    tag = f"prefix[{impl_name}:{gate}:P{P}]"
    leaks = {}
    if not torch.equal(base.o[:, :P], pt.o[:, :P]):
        leaks["output"] = _leak(base.o[:, :P], pt.o[:, :P])
    if not bwd:
        pass
    elif all(torch.equal(base.grads[n], base2.grads[n]) for n in base.grads):
        for n in TOKEN_LEAVES:  # A_log / dt_bias are reductions over all tokens, not compared
            if n in base.grads and not torch.equal(base.grads[n][:, :P], pt.grads[n][:, :P]):
                leaks[f"d{n}"] = _leak(base.grads[n][:, :P], pt.grads[n][:, :P])
    elif not leaks:
        pytest.skip("backward is not run-to-run deterministic; prefix-grad bitwise check is meaningless")
    record(test=tag, leaks=leaks)
    assert not leaks, f"{tag}: prefix changed when future tokens were perturbed: {leaks}"


def _leak(a, b):
    """{max abs diff, first differing token position} of two [B, P, ...] prefix tensors."""
    d = (a.double() - b.double()).abs().transpose(0, 1).reshape(a.shape[1], -1).amax(1)
    return {"max_abs_diff": d.max().item(), "first_pos": int((d > 0).nonzero()[0])}


# ---------------------------------------------------------------------------
# 3. document boundary (varlen)
# ---------------------------------------------------------------------------

DOC_CASE = Case(gate="kimi", beta="s2", gva=True, layout="varlen", l2=True, h0=True)


@pytest.mark.gpu
@impl_param
@pytest.mark.parametrize("doc", [0, 2, 4])
def test_doc_boundary(impl_name, doc):
    """Perturbing one document leaves every other document's output, final state and grads bitwise unchanged."""
    impl = get_impl(impl_name)
    inp, other = make_inputs(DOC_CASE, seed=0), make_inputs(DOC_CASE, seed=1)
    s, e = inp.dims.cu[doc], inp.dims.cu[doc + 1]

    def splice(n, t):
        if n in TOKEN_LEAVES:
            return torch.cat([t[:, :s], other.leaves[n][:, s:e], t[:, e:]], 1)
        return torch.cat([t[:doc], other.leaves[n][doc:doc + 1], t[doc + 1:]]) if n == "h0" else t

    pert = dataclasses.replace(inp, leaves={n: splice(n, t) for n, t in inp.leaves.items()})
    run = make_run(DOC_CASE)
    bwd = impl.supports_bwd
    base, base2, pt = (run_impl(impl, i, run, backward=bwd) for i in (inp, inp, pert))
    keep = torch.ones(inp.dims.T, dtype=torch.bool, device=inp.do.device)
    keep[s:e] = False
    docs = [d for d in range(inp.dims.N) if d != doc]
    assert torch.equal(base.o[:, keep], pt.o[:, keep]), "output of another document changed"
    assert torch.equal(base.ht[docs], pt.ht[docs]), "final state of another document changed"
    if bwd and all(torch.equal(base.grads[n], base2.grads[n]) for n in base.grads):
        for n in base.grads:
            if n in TOKEN_LEAVES:
                assert torch.equal(base.grads[n][:, keep], pt.grads[n][:, keep]), f"d{n} of another document changed"
            elif n == "h0":
                assert torch.equal(base.grads[n][docs], pt.grads[n][docs]), "dh0 of another document changed"


@pytest.mark.gpu
@impl_param
def test_varlen_equals_per_document(impl_name):
    """One varlen call == separate per-document calls (metric thresholds), outputs, final states and grads."""
    impl = get_impl(impl_name)
    inp = make_inputs(DOC_CASE)
    run = make_run(DOC_CASE)
    bwd = impl.supports_bwd
    full = run_impl(impl, inp, run, backward=bwd)
    parts = [run_impl(impl, doc_slice(inp, n), run, backward=bwd) for n in range(inp.dims.N)]
    o = torch.cat([p.o for p in parts], 1)
    ht = torch.cat([p.ht for p in parts], 0)
    if not bwd:
        _check_fwd(f"varlen_vs_docs[{impl_name}]", full, dataclasses.replace(full, o=o, ht=ht))
        return
    grads = {n: torch.cat([p.grads[n] for p in parts], 1 if n in TOKEN_LEAVES else 0)
             for n in parts[0].grads if n in TOKEN_LEAVES or n == "h0"}
    sep = dataclasses.replace(full, o=o, ht=ht, grads={**full.grads, **grads})  # A_log / dt_bias: sum of parts
    for n in ("A_log", "dt_bias"):
        sep.grads[n] = sum(p.grads[n] for p in parts)
    _check_fwd(f"varlen_vs_docs[{impl_name}]", full, dataclasses.replace(sep, grads=None))
    errs = {n: rel_err(full.grads[n], sep.grads[n])[0] for n in sep.grads}
    record(test=f"varlen_vs_docs_grads[{impl_name}]", grads=errs)
    assert max(errs.values()) < THRESH["grad_max"], errs


# ---------------------------------------------------------------------------
# 4. GVA: HV = 2 * HK == running with q/k repeat_interleaved to HV heads
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@impl_param
@pytest.mark.parametrize("layout", ["batch", "varlen"])
def test_gva(impl_name, layout):
    impl = get_impl(impl_name)
    case = Case(gate="kimi", beta="s2", gva=True, layout=layout, l2=True, h0=True)
    inp = make_inputs(case)
    G = inp.dims.HV // inp.dims.HK
    exp = dataclasses.replace(inp, dims=inp.dims._replace(HK=inp.dims.HV), leaves={
        **inp.leaves, "q": inp.leaves["q"].repeat_interleave(G, 2), "k": inp.leaves["k"].repeat_interleave(G, 2)})
    run = make_run(case)
    bwd = impl.supports_bwd
    a, b = run_impl(impl, inp, run, backward=bwd), run_impl(impl, exp, run, backward=bwd)
    _check_fwd(f"gva[{impl_name}:{layout}]", a, b)
    if not bwd:
        return
    B, T, HV, D = inp.leaves["v"].shape
    for n in ("q", "k"):  # grads of the expanded copies must reduce to the shared-head grads
        b.grads[n] = b.grads[n].double().reshape(B, T, HV // G, G, D).sum(3)
    _check_grads(f"gva_grads[{impl_name}:{layout}]", a, b)


# ---------------------------------------------------------------------------
# 5. tiny shapes: kernel grads vs fp64 reference autograd (bf16 stand-in for gradcheck)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@impl_param
@pytest.mark.parametrize("T, HK, HV, D", [(64, 1, 1, 128), (100, 1, 2, 128), (130, 2, 2, 128), (130, 1, 2, 64)])
def test_grad_small_vs_ref(impl_name, T, HK, HV, D):
    impl = get_impl(impl_name)
    if D != 128 and impl_name != "fla":
        pytest.skip("first release: Dk = Dv = 128")
    case = Case(gate="kimi", beta="s2", gva=HV > HK, l2=True, epi="silu", h0=True)
    inp = make_inputs(case, dims_for(case, B=1, T=T, HK=HK, HV=HV, D=D))
    run = make_run(case)
    bwd = impl.supports_bwd
    res, ref = run_impl(impl, inp, run, backward=bwd), run_ref(inp, run, backward=bwd)
    _check_fwd(f"small_fwd[{impl_name}:T{T}H{HK}/{HV}D{D}]", res, ref)
    if not bwd:
        return
    _check_grads(f"small_bwd[{impl_name}:T{T}H{HK}/{HV}D{D}]", res, ref)


# ---------------------------------------------------------------------------
# 6. state layout and determinism
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@impl_param
def test_state_v_first(impl_name):
    """[N, HV, Dv, Dk] state layout gives the same o / final state as [Dk, Dv]."""
    impl = get_impl(impl_name)
    case = Case(gate="kimi", beta="s2", gva=True, layout="varlen", l2=True, h0=True)
    inp = make_inputs(case)
    try:
        a = run_impl(impl, inp, make_run(case), backward=False)
        b = run_impl(impl, inp, make_run(case, state_v_first=True), backward=False)
    except ImplUnavailable as e:
        pytest.skip(str(e))
    _check_fwd(f"state_v_first[{impl_name}]", b, a)


@pytest.mark.gpu
@pytest.mark.slow
@impl_param
@pytest.mark.parametrize("case", [Case(gate="kimi", gva=True, h0=True), Case(gate="eda", layout="varlen", h0=True)],
                         ids=lambda c: c.id)
def test_fwd_deterministic(impl_name, case):
    impl = get_impl(impl_name)
    inp, run = make_inputs(case), make_run(case)
    first = run_impl(impl, inp, run, backward=False)
    for i in range(20):
        r = run_impl(impl, inp, run, backward=False)
        assert torch.equal(r.o, first.o), f"o differs at iteration {i}"
        assert torch.equal(r.ht, first.ht), f"final state differs at iteration {i}"
