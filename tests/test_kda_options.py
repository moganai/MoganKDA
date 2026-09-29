"""In-kernel options of ``chunk_delta_rule`` (gate, beta sigmoid, L2 norm, epilogue) vs the torch reference of the options.

* new path (kernels, ``ChunkDeltaRuleKDAFunction``) vs ``MOGAN_KDA_EMULATE=1`` (torch options around the plain
  kernels): forward and gradients of every leaf, on a few option combinations
* epilogue kernel vs torch (fp32 / bf16, both acts, row count not a multiple of the tile)
* timing (run with -s): full fwd+bwd, eda gate + sigmoid beta x2 + silu epilogue, new vs emulation vs FLA chunk_kda
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kda_common import (  # noqa: E402
    Case, ImplUnavailable, dims_for, load_impl, make_inputs, make_run, rel_err, run_impl, Run,
)
from ref_kda import ref_output_norm  # noqa: E402

DEVICE = "cuda"
FWD_TOL, GRAD_TOL = 2e-2, 5e-2  # max relative to max|ref|; new and emulation differ only by bf16 rounding places

COMBOS = [
    pytest.param(Case("eda", "s2", True, "varlen", True, "silu", True), id="eda-s2-gva-varlen-l2-silu-h0"),
    pytest.param(Case("kimi", "p1", False, "batch", False, "sigmoid", False), id="kimi-p1-batch-sigmoid"),
    pytest.param(Case("gdn", "s1", True, "varlen", True, "none", False), id="gdn-s1-gva-varlen-l2"),
    pytest.param(Case("precomputed", "s2", False, "batch", True, "silu", True), id="precomputed-s2-batch-l2-silu-h0"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("case", COMBOS)
def test_new_vs_emulation(case, monkeypatch):
    from arch_utils import ARCH
    if ARCH != "SM90":
        pytest.skip("Hopper only")
    impl = load_impl("qla")
    inp, run = make_inputs(case), make_run(case)
    res = {}
    for name, emu in (("new", "0"), ("emu", "1")):
        monkeypatch.setenv("MOGAN_KDA_EMULATE", emu)
        res[name] = run_impl(impl, inp, run, backward=True)
    a, b = res["new"], res["emu"]
    fm, _ = rel_err(a.o, b.o)
    assert fm < FWD_TOL, f"fwd max rel err {fm:.3e}"
    if a.ht is not None:
        assert rel_err(a.ht, b.ht)[0] < FWD_TOL
    for n, gb in b.grads.items():
        ga = a.grads[n]
        assert ga is not None, f"missing grad {n}"
        e = rel_err(ga, gb)[0]
        assert e < GRAD_TOL, f"grad {n}: max rel err {e:.3e}"


@pytest.mark.gpu
@pytest.mark.parametrize("act", ["silu", "sigmoid"])
@pytest.mark.parametrize("o_dtype", [torch.float32, torch.bfloat16], ids=["o-fp32", "o-bf16"])
@pytest.mark.parametrize("g_dtype", [torch.float32, torch.bfloat16], ids=["g-fp32", "g-bf16"])
@pytest.mark.parametrize("with_w", [True, False], ids=["w", "now"])
def test_epilogue_kernel(act, o_dtype, g_dtype, with_w):
    from mogan_kda.ops.kimi_delta_rule.epilogue import rmsnorm_gated_fwd, rmsnorm_gated_bwd

    torch.manual_seed(0)
    shape = (1, 37, 3, 128)  # 111 rows: not a multiple of the 16-row tile
    o = (torch.randn(shape, device=DEVICE) * 2).to(o_dtype)
    gate = torch.randn(shape, device=DEVICE).to(g_dtype)
    w = (1.0 + 0.1 * torch.randn(128, device=DEVICE)) if with_w else None
    dy = torch.randn(shape, device=DEVICE)
    y = rmsnorm_gated_fwd(o, w, gate, act)
    do, dw, dgate = rmsnorm_gated_bwd(dy, o, w, gate, act)

    o64 = o.double().requires_grad_()
    g64 = gate.double().requires_grad_()
    w64 = None if w is None else w.double().requires_grad_()
    y64 = ref_output_norm(o64, w64, g64, act)
    y64.backward(dy.double())
    tight = o_dtype == g_dtype == torch.float32
    tol = 2e-5 if tight else 1e-2
    for name, got, ref in (("y", y, y64.detach()), ("do", do, o64.grad), ("dgate", dgate, g64.grad)):
        assert got.dtype == {"y": torch.float32, "do": o_dtype, "dgate": g_dtype}[name]
        e = rel_err(got, ref)[0]
        assert e < (2e-5 if (tight or name == "y") else tol), f"{name}: {e:.3e}"
    if with_w:
        assert rel_err(dw, w64.grad)[0] < 1e-4
    else:
        assert dw is None
    do2, dw2, dgate2 = rmsnorm_gated_bwd(dy, o, w, gate, act)
    assert torch.equal(do, do2) and torch.equal(dgate, dgate2)
    assert (dw is None) or torch.equal(dw, dw2), "dw not deterministic"


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def _time(fn, iters=10):
    for _ in range(2):
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
def test_timing_full(monkeypatch):
    """[1, 16384, HK=8, HV=16, 128], 19 packed docs; eda + sigmoid beta x2 + silu epilogue (FLA: kimi gate in kernel,
    epilogue emulated in torch: FLA has no eda / epilogue)."""
    from arch_utils import ARCH
    if ARCH != "SM90":
        pytest.skip("Hopper only")
    T_, HK, HV, D = 16384, 8, 16, 128
    g = torch.Generator().manual_seed(0)
    cuts = torch.sort(torch.randperm(T_ - 1, generator=g)[:18] + 1).values.tolist()
    cu_list = [0] + cuts + [T_]
    dims = dims_for(Case("eda", "s2", True, "varlen", True, "silu", False))._replace(
        B=1, T=T_, HK=HK, HV=HV, D=D, cu=tuple(cu_list))
    case = Case("eda", "s2", True, "varlen", True, "silu", False)
    inp = make_inputs(case, dims)
    run = make_run(case)
    do = inp.do

    def make_fn(impl, emu=None):
        def fn():
            if emu is not None:
                os.environ["MOGAN_KDA_EMULATE"] = emu
            lv = {n: t.detach().clone().requires_grad_() for n, t in inp.leaves.items()}
            o, _, _ = impl(lv, inp.cu, run)
            torch.autograd.grad((o.float() * do).sum(), list(lv.values()), allow_unused=True)
        return fn

    old = os.environ.get("MOGAN_KDA_EMULATE")
    try:
        qla = load_impl("qla")
        t_new = _time(make_fn(qla, "0"))
        t_emu = _time(make_fn(qla, "1"))
    finally:
        if old is None:
            os.environ.pop("MOGAN_KDA_EMULATE", None)
        else:
            os.environ["MOGAN_KDA_EMULATE"] = old
    msg = f"new {t_new:.0f} us | emulation {t_emu:.0f} us"
    try:
        fla = load_impl("fla")
        fla_run = Run("kimi", -5.0, run.beta_mode, run.beta_scale, run.l2, run.epi, False)
        msg += f" | FLA (kimi gate, torch epilogue) {_time(make_fn_fla(fla, fla_run, inp, do)):.0f} us"
    except ImplUnavailable as e:
        msg += f" | FLA n/a ({e})"
    print(f"\n[full fwd+bwd, eda+sigmoid-beta x2+silu epilogue, [1,{T_},{HK},{HV},{D}] 19 docs] {msg}")


def make_fn_fla(impl, run, inp, do):
    def fn():
        lv = {n: t.detach().clone().requires_grad_() for n, t in inp.leaves.items()}
        o, _, _ = impl(lv, inp.cu, run)
        torch.autograd.grad((o.float() * do).sum(), list(lv.values()), allow_unused=True)
    return fn
