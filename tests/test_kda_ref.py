"""CPU self-consistency tests of the fp64 reference ``ref_kda.py`` (no GPU needed).

Independent oracles: FLA's naive recurrences (``_ref/flash-linear-attention``, converted to fp64; tests are
skipped when the checkout or einops is missing), the closed-form gate formulas, mixers.py (optional) and
hand-written epilogue / invariance checks. Runs in a few seconds.
"""
import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from ref_kda import ref_beta, ref_chunk_delta_rule, ref_gate, ref_kda_fla_compat, ref_output_norm

F64 = torch.float64
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLA_OPS = os.path.join(REPO, "_ref", "flash-linear-attention", "fla", "ops")
KOD = os.environ.get("KDA_KOD", os.path.expanduser("~/Desktop/foundationdata/LLM_ablation/kod"))


def _load_fp64(path, name):
    """Import a FLA naive module with its float32 casts turned into float64."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    pytest.importorskip("einops")
    src = open(path).read()
    for a, b in (("torch.float)", "torch.float64)"), ("torch.float]", "torch.float64]"),
                 ("torch.float32", "torch.float64"), ("dtype=torch.float,", "dtype=torch.float64,"),
                 (".float()", ".double()")):
        src = src.replace(a, b)
    m = types.ModuleType(name)
    exec(compile(src, path, "exec"), m.__dict__)
    return m


@pytest.fixture(scope="module")
def fla_kda():
    return _load_fp64(os.path.join(FLA_OPS, "kda", "naive.py"), "fla_kda_naive")


@pytest.fixture(scope="module")
def fla_gdn():
    return _load_fp64(os.path.join(FLA_OPS, "gated_delta_rule", "naive.py"), "fla_gdn_naive")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make(B=2, T=128, HK=2, HV=4, D=32, channel=True):
    q = F.normalize(torch.randn(B, T, HK, D, dtype=F64), dim=-1)
    k = F.normalize(torch.randn(B, T, HK, D, dtype=F64), dim=-1)
    v = torch.randn(B, T, HV, D, dtype=F64)
    g = -5 * torch.sigmoid(torch.randn(B, T, HV, D, dtype=F64))
    beta = 2 * torch.sigmoid(torch.randn(B, T, HV, dtype=F64))
    h0 = torch.randn(B, HV, D, D, dtype=F64) * 0.1
    return q, k, v, (g if channel else g[..., 0]), beta, h0


mx = lambda a, b: (a - b).abs().max().item()  # noqa: E731


# ---------------------------------------------------------------------------
# vs FLA naive
# ---------------------------------------------------------------------------

def test_vs_fla_naive_recurrent_and_chunk(fla_kda):
    q, k, v, g, beta, h0 = make()
    o, s = ref_chunk_delta_rule(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    o1, s1 = fla_kda.naive_recurrent_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    assert mx(o, o1) < 1e-12 and mx(s, s1) < 1e-12
    o2, s2 = fla_kda.naive_chunk_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    assert mx(o, o2) < 1e-9 and mx(s, s2) < 1e-9


def test_scalar_gate_vs_fla_gdn(fla_gdn):
    q, k, v, g, beta, h0 = make(HK=4, HV=4, channel=False)
    o, s = ref_chunk_delta_rule(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    o1, s1 = fla_gdn.naive_recurrent_gated_delta_rule(q, k, v, beta, g, initial_state=h0, output_final_state=True)
    assert mx(o, o1) < 1e-12 and mx(s, s1) < 1e-12


def test_scalar_equals_expanded_channel_gate():
    q, k, v, g, beta, _ = make(channel=False)
    a, _ = ref_chunk_delta_rule(q, k, v, g, beta)
    b, _ = ref_chunk_delta_rule(q, k, v, g[..., None].expand(*g.shape, q.shape[-1]), beta)
    assert mx(a, b) == 0.0


# ---------------------------------------------------------------------------
# gates / beta / epilogue
# ---------------------------------------------------------------------------

def test_gate_forms_match_formulas():
    a = torch.randn(2, 16, 4, 32, dtype=F64) * 3
    dtb, A_log = torch.randn(4, 32, dtype=F64), torch.randn(4, dtype=F64)
    A = A_log.exp()[:, None]
    assert mx(ref_gate(a, "gdn", A_log, dtb, None), -A * F.softplus(a + dtb)) < 1e-12        # fla naive_kda_gate
    assert mx(ref_gate(a, "gdn", A_log, dtb, -1.0), (-A * F.softplus(a + dtb)).clamp(min=-1.0)) < 1e-12
    assert mx(ref_gate(a, "kimi", A_log, dtb, -5.0), -5 * torch.sigmoid(A * (a + dtb))) < 1e-12
    assert mx(ref_gate(a, "kimi", None, dtb, -5.0), -5 * torch.sigmoid(a + dtb)) < 1e-12      # A_log optional
    eda = -5 + 5 * torch.exp(-(A / 5) * F.softplus(a + dtb))                                  # mixers.safe_gate
    assert mx(ref_gate(a, "eda", A_log, dtb, -5.0), eda) < 1e-12
    for m in ("eda", "kimi"):
        g = ref_gate(a, m, A_log, dtb, -5.0)
        assert g.max() < 0 and g.min() >= -5.0
    a3, d3 = torch.randn(2, 16, 4, dtype=F64), torch.randn(4, dtype=F64)                      # scalar gate
    assert mx(ref_gate(a3, "gdn", A_log, d3, None), -A_log.exp() * F.softplus(a3 + d3)) < 1e-12


def test_gate_forms_match_mixers():
    if not os.path.exists(os.path.join(KOD, "mixers.py")):
        pytest.skip("training code (mixers.py) not found")
    sys.path.insert(0, KOD)
    try:
        spec = importlib.util.spec_from_file_location("mixers_kod", os.path.join(KOD, "mixers.py"))
        MX = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MX)
    finally:
        sys.path.remove(KOD)
    a = torch.randn(2, 16, 4, 32, dtype=F64) * 3
    dtb, A_log = torch.randn(4, 32, dtype=F64), torch.randn(4, dtype=F64)
    # mixers.py computes the gates in fp32 -> fp32 tolerance
    assert mx(ref_gate(a, "eda", A_log, dtb, -5.0), MX.safe_gate(a, dtb, A_log).double()) < 1e-5
    assert mx(ref_gate(a, "kimi", A_log, dtb, -5.0), MX.kimi_gate(a, dtb, A_log).double()) < 1e-5


def test_beta_modes():
    b = torch.randn(2, 8, 4, dtype=F64)
    assert mx(ref_beta(b, "precomputed"), b) == 0.0
    assert mx(ref_beta(b, "sigmoid", 2.0), 2 * torch.sigmoid(b)) < 1e-15


@pytest.mark.parametrize("act", ["silu", "sigmoid"])
def test_output_epilogue(act):
    q, k, v, g, beta, _ = make(B=1, T=8)
    w, gate = torch.randn(32, dtype=F64), torch.randn(1, 8, 4, 32, dtype=F64)
    o, _ = ref_chunk_delta_rule(q, k, v, g, beta)
    o_ep, _ = ref_chunk_delta_rule(q, k, v, g, beta, output_norm=dict(weight=w, gate=gate, act=act))
    a = F.silu(gate) if act == "silu" else torch.sigmoid(gate)
    assert mx(o_ep, o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * w * a) < 1e-12
    assert mx(ref_output_norm(o, None, gate, act), o_ep / w) < 1e-12  # weight=None == weight 1


# ---------------------------------------------------------------------------
# structure: prefix invariance, varlen, state layout, GVA
# ---------------------------------------------------------------------------

def test_prefix_invariance_exact():
    q, k, v, g, beta, _ = make(B=1, T=96)
    P = 41

    def run(perturb):
        xs = [t.clone() for t in (q, k, v, g, beta)]
        if perturb:
            for t in xs:
                t[:, P:] = torch.randn_like(t[:, P:]) * 7
        xs = [t.requires_grad_() for t in xs]
        o, _ = ref_chunk_delta_rule(*xs, use_qk_l2norm_in_kernel=True)
        o[:, :P].sum().backward()
        return o[:, :P].detach(), [t.grad[:, :P] for t in xs]

    (o1, g1), (o2, g2) = run(False), run(True)
    assert torch.equal(o1, o2)
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_varlen_equals_per_segment_with_states(fla_kda):
    q, k, v, g, beta, _ = make(B=1, T=100)
    cu = torch.tensor([0, 17, 64, 65, 100])
    h0 = torch.randn(4, 4, 32, 32, dtype=F64)
    o, s = ref_chunk_delta_rule(q, k, v, g, beta, cu_seqlens=cu, initial_state=h0, output_final_state=True)
    for n in range(4):
        a, b = cu[n].item(), cu[n + 1].item()
        o_n, s_n = fla_kda.naive_recurrent_kda(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b],
                                               initial_state=h0[n:n + 1], output_final_state=True)
        assert mx(o[:, a:b], o_n) < 1e-12 and mx(s[n:n + 1], s_n) < 1e-12


def test_varlen_equals_per_segment_own_ref():  # no FLA needed
    q, k, v, g, beta, _ = make(B=1, T=60)
    cu = torch.tensor([0, 1, 33, 60])
    o, s = ref_chunk_delta_rule(q, k, v, g, beta, cu_seqlens=cu, output_final_state=True)
    for n in range(3):
        a, b = cu[n].item(), cu[n + 1].item()
        o_n, s_n = ref_chunk_delta_rule(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], output_final_state=True)
        assert mx(o[:, a:b], o_n) == 0.0 and mx(s[n:n + 1], s_n) == 0.0


def test_state_v_first():
    q, k, v, g, beta, _ = make(B=1, T=100)
    cu = torch.tensor([0, 17, 64, 65, 100])
    h0 = torch.randn(4, 4, 32, 32, dtype=F64)
    o, s = ref_chunk_delta_rule(q, k, v, g, beta, cu_seqlens=cu, initial_state=h0, output_final_state=True)
    o2, s2 = ref_chunk_delta_rule(q, k, v, g, beta, cu_seqlens=cu, initial_state=h0.transpose(-1, -2),
                                  output_final_state=True, state_v_first=True)
    assert mx(o, o2) == 0.0 and mx(s, s2.transpose(-1, -2)) == 0.0


def test_gva_equals_expanded_heads():
    q, k, v, g, beta, _ = make(HK=2, HV=4)
    a, _ = ref_chunk_delta_rule(q, k, v, g, beta)
    b, _ = ref_chunk_delta_rule(q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, g, beta)
    assert mx(a, b) == 0.0


# ---------------------------------------------------------------------------
# autograd / compat wrapper
# ---------------------------------------------------------------------------

def test_gradcheck_all_options():
    B, T, HK, HV, D = 1, 6, 1, 2, 4
    r = lambda *s: torch.randn(*s, dtype=F64)  # noqa: E731
    ts = [r(B, T, HK, D), r(B, T, HK, D), r(B, T, HV, D), r(B, T, HV, D), r(B, T, HV), r(B, HV, D, D) * 0.1,
          r(HV) * 0.3, r(HV, D), r(B, T, HV, D), r(D)]
    ts = [t.requires_grad_() for t in ts]

    def f(q, k, v, g, b, h0, A_log, dtb, gate, w):
        return ref_chunk_delta_rule(
            q, k, v, g, b, initial_state=h0, output_final_state=True, gate_mode="eda", A_log=A_log, dt_bias=dtb,
            beta_mode="sigmoid", beta_scale=2.0, use_qk_l2norm_in_kernel=True,
            output_norm=dict(weight=w, gate=gate, act="silu"))

    assert torch.autograd.gradcheck(f, ts, eps=1e-6, atol=1e-7)


def test_fla_compat_wrapper(fla_kda):
    q, k, v, g, beta, _ = make(B=1, T=32)
    dtb, A_log = torch.randn(4, 32, dtype=F64), torch.randn(4, dtype=F64)
    o, _ = ref_kda_fla_compat(q, k, v, g, beta, use_gate_in_kernel=True, A_log=A_log, dt_bias=dtb.flatten(),
                              lower_bound=-5.0, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    gk = -5 * torch.sigmoid(A_log.exp()[:, None] * (g + dtb))
    o1, _ = fla_kda.naive_recurrent_kda(q, k, v, gk, 2 * torch.sigmoid(beta))
    assert mx(o, o1) < 1e-12
    o2, _ = ref_kda_fla_compat(q, k, v, g, beta, use_gate_in_kernel=True, A_log=A_log, dt_bias=dtb.flatten())
    g2 = -A_log.exp()[:, None] * F.softplus(g + dtb)                       # no lower_bound -> softplus form
    o3, _ = fla_kda.naive_recurrent_kda(q, k, v, g2, beta)
    assert mx(o2, o3) < 1e-12
    with pytest.raises(NotImplementedError):
        ref_kda_fla_compat(q, k, v, g, beta, cp_context=object())
