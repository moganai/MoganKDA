"""Shared scaffolding for the KDA / unified delta-rule tests (``test_kda_unit.py``).

* option matrix (``Case``), reduced pairwise matrix and full matrix
* realistic input generation (bf16 q/k/v, fp32 gate/beta, per-mode gate parameters)
* implementation adapters: ``fla`` (``fla.ops.kda.chunk_kda``, second reference) and
  ``qla`` (our mogan_kda entry point). An adapter maps the unified option set onto the
  implementation's kwargs (``native``) or applies the option outside in torch (``emulated``).
* the training-test metric (same as ``kod/test_mixers.py``) and the fp64 reference runner.

No fla / mogan_kda import at module level (the CPU tests import ``ref_kda`` only).
"""
import dataclasses
import functools
import inspect
import itertools
import json
import os
import sys
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_kda import ref_chunk_delta_rule, ref_output_norm  # noqa: E402

DEVICE = "cuda"
LOWER_BOUND = -5.0
# Same thresholds / metric as the training tests (LLM_ablation/kod/test_mixers.py).
THRESH = {"fwd_max": 2e-2, "fwd_mean": 5e-3, "grad_max": 5e-2}
BETAS = {"p1": ("precomputed", 1.0), "p2": ("precomputed", 2.0), "s1": ("sigmoid", 1.0), "s2": ("sigmoid", 2.0)}
TOKEN_LEAVES = ("q", "k", "v", "g", "beta", "ep_gate")  # leaves with a [B, T, ...] token axis


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def rel_err(a, b):
    """(max|a-b| / max|b|, mean|a-b| / mean|b|) in fp64."""
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return (d.max() / b.abs().max().clamp_min(1e-12)).item(), (d.mean() / b.abs().mean().clamp_min(1e-12)).item()


def grad_errs(res, ref):
    """{leaf name: max rel err}; a missing gradient counts as inf."""
    return {n: (rel_err(res.grads[n], gr)[0] if res.grads.get(n) is not None else float("inf"))
            for n, gr in ref.grads.items()}


# ---------------------------------------------------------------------------
# Option matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    gate: str = "kimi"          # precomputed | eda | kimi | gdn
    beta: str = "s2"            # p1 / p2: precomputed in (0,1)/(0,2); s1 / s2: raw logits, sigmoid * 1 / 2
    gva: bool = False           # HV = 2 * HK
    layout: str = "batch"       # batch (B=2) | varlen (B=1 + random docs incl. length-1 and non-multiple-of-64)
    l2: bool = False            # L2-normalize q/k inside the kernel
    epi: str = "none"           # none | silu | sigmoid (per-head RMSNorm * act(gate))
    h0: bool = False            # initial state (+ final state and dht)

    @property
    def id(self):
        return "-".join([self.gate, self.beta, "gva" if self.gva else "mha", self.layout,
                         "l2" if self.l2 else "nol2", self.epi, "h0" if self.h0 else "noh0"])


FACTORS = {
    "gate": ["precomputed", "eda", "kimi", "gdn"], "beta": list(BETAS), "gva": [False, True],
    "layout": ["batch", "varlen"], "l2": [False, True], "epi": ["none", "silu", "sigmoid"], "h0": [False, True],
}


def full_matrix():
    return [Case(*c) for c in itertools.product(*FACTORS.values())]


def reduced_matrix():
    """Greedy all-pairs covering list: every pair of option values appears in at least one case."""
    cases = list(itertools.product(*FACTORS.values()))
    n = len(FACTORS)
    pairs = lambda c: {((i, c[i]), (j, c[j])) for i in range(n) for j in range(i + 1, n)}  # noqa: E731
    uncovered = set().union(*(pairs(c) for c in cases))
    chosen = []
    while uncovered:
        best = max(cases, key=lambda c: len(pairs(c) & uncovered))
        chosen.append(best)
        uncovered -= pairs(best)
    return [Case(*c) for c in chosen]


class Dims(NamedTuple):
    B: int
    T: int
    HK: int
    HV: int
    D: int
    cu: tuple | None  # document boundaries (varlen) or None

    @property
    def N(self):
        return len(self.cu) - 1 if self.cu else self.B


VARLEN_CU = (0, 1, 98, 354, 504, 704)  # docs of length 1, 97, 256, 150, 200


def dims_for(case, **over):
    d = dict(B=1, T=704, cu=VARLEN_CU) if case.layout == "varlen" else dict(B=2, T=300, cu=None)
    d.update(HK=2, HV=4 if case.gva else 2, D=128)
    d.update(over)
    return Dims(**d)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    leaves: dict          # name -> tensor (differentiable inputs, detached originals)
    do: torch.Tensor      # fp32 [B, T, HV, Dv]
    dht: torch.Tensor | None
    cu: torch.Tensor | None
    dims: Dims


def make_inputs(case, dims=None, seed=0, dev=DEVICE):
    """Realistic data: L2-normalized q/k (raw when the kernel normalizes), g in [-5, 0), beta per mode,
    gate parameters like training (A_log = log U(0.2, 4), dt_bias ~ N(-1, 0.5))."""
    dims = dims or dims_for(case)
    B, T, HK, HV, D = dims.B, dims.T, dims.HK, dims.HV, dims.D
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32)  # noqa: E731
    q, k = r(B, T, HK, D), r(B, T, HK, D)
    if not case.l2:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    lv = {"q": q.bfloat16(), "k": k.bfloat16(), "v": r(B, T, HV, D).bfloat16()}
    if case.gate == "precomputed":
        lv["g"] = LOWER_BOUND * torch.sigmoid(r(B, T, HV, D))
    else:
        lv["g"] = r(B, T, HV, D)
        # gdn form: -exp(A_log) softplus(x) is unbounded below; keep the per-token decay >= -5 (the contract of the
        # anchored kernels) with a smaller amplitude range. eda / kimi are bounded by construction.
        a_hi = 0.8 if case.gate == "gdn" else 4.0
        lv["A_log"] = torch.empty(HV, device=dev).uniform_(min(0.2, a_hi / 4), a_hi).log()
        lv["dt_bias"] = r(HV, D) * 0.5 - 1.0
    mode, scale = BETAS[case.beta]
    lv["beta"] = r(B, T, HV) if mode == "sigmoid" else scale * torch.sigmoid(r(B, T, HV))
    if case.h0:
        lv["h0"] = r(dims.N, HV, D, D) * 0.1
    if case.epi != "none":
        lv["ep_w"] = 1.0 + 0.1 * r(D)
        lv["ep_gate"] = r(B, T, HV, D)
    dht = r(dims.N, HV, D, D) * 0.1 if case.h0 else None
    cu = torch.tensor(dims.cu, device=dev, dtype=torch.int32) if dims.cu else None
    return Inputs(lv, r(B, T, HV, D), dht, cu, dims)


def doc_slice(inp, n):
    """Inputs of document n of a varlen batch as a stand-alone (cu=None) problem."""
    s, e = inp.dims.cu[n], inp.dims.cu[n + 1]
    lv = {nm: (t[:, s:e] if nm in TOKEN_LEAVES else (t[n:n + 1] if nm == "h0" else t)) for nm, t in inp.leaves.items()}
    dims = inp.dims._replace(B=1, T=e - s, cu=None)
    return Inputs(lv, inp.do[:, s:e], None if inp.dht is None else inp.dht[n:n + 1], None, dims)


# ---------------------------------------------------------------------------
# Run configuration and result
# ---------------------------------------------------------------------------

@dataclass
class Run:
    gate_mode: str
    lower_bound: float | None
    beta_mode: str
    beta_scale: float
    l2: bool
    epi: str | None
    final_state: bool
    state_v_first: bool = False
    safe_gate: bool = False   # FLA kernel path selector only (M=16 TensorCore); math identical to "kimi"


def make_run(case, state_v_first=False, safe_gate=False):
    bm, bs = BETAS[case.beta]
    lb = LOWER_BOUND if case.gate in ("eda", "kimi") else None
    return Run(case.gate, lb, "sigmoid" if bm == "sigmoid" else "precomputed", bs, case.l2,
               None if case.epi == "none" else case.epi, case.h0, state_v_first, safe_gate)


@dataclass
class Result:
    o: torch.Tensor
    ht: torch.Tensor | None
    grads: dict | None
    notes: dict
    o_pre: torch.Tensor | None = None  # reference only: output before the epilogue (fp64), set when run.epi


def _finish(leaves, o, ht, inp, notes, backward=True, o_pre=None):
    if not backward:
        return Result(o.detach(), None if ht is None else ht.detach(), None, notes, o_pre)
    loss = (o.double() * inp.do.double()).sum()
    if ht is not None and inp.dht is not None:
        loss = loss + (ht.double() * inp.dht.double()).sum()
    names = list(leaves)
    gs = torch.autograd.grad(loss, [leaves[n] for n in names], allow_unused=True)
    return Result(o.detach(), None if ht is None else ht.detach(), dict(zip(names, gs)), notes, o_pre)


def run_ref(inp, run, backward=True):
    """fp64 reference (``ref_chunk_delta_rule``, always [K, V] state layout)."""
    lv = {n: t.detach().double().requires_grad_() for n, t in inp.leaves.items()}
    o, ht = ref_chunk_delta_rule(
        lv["q"], lv["k"], lv["v"], lv["g"], lv["beta"], initial_state=lv.get("h0"),
        output_final_state=run.final_state, cu_seqlens=inp.cu, gate_mode=run.gate_mode,
        A_log=lv.get("A_log"), dt_bias=lv.get("dt_bias"), lower_bound=run.lower_bound,
        beta_mode=run.beta_mode, beta_scale=run.beta_scale, use_qk_l2norm_in_kernel=run.l2)
    o_pre = None
    if run.epi:  # same as ref_chunk_delta_rule(output_norm=...), but keeps the pre-epilogue output
        o_pre = o.detach()
        o = ref_output_norm(o, lv["ep_w"], lv["ep_gate"], run.epi)
    return _finish(lv, o, ht, inp, {}, backward, o_pre)


def epi_weighted_err(res, ref):
    """Error of an epilogue output measured on the pre-norm scale: rel_err(o * w, o_ref * w) with
    w[b,t,h] = per-(token, head) RMS of the reference's pre-epilogue output. RMSNorm is scale-invariant, so a
    bf16 absolute error of the kernel at a near-zero-output token is amplified by the per-token normalization
    (raw error there is a metric artifact); weighting by the reference RMS undoes that. A wrong weight / gate /
    act still gives an O(1) error. Returns (max, mean)."""
    w = torch.sqrt(ref.o_pre.pow(2).mean(-1, keepdim=True) + 1e-6)
    return rel_err(res.o.double() * w, ref.o * w)


def run_impl(impl, inp, run, backward=True):
    lv = {n: t.detach().clone().requires_grad_(backward) for n, t in inp.leaves.items()}
    x = dict(lv)
    if run.state_v_first and "h0" in x:
        x["h0"] = lv["h0"].transpose(-1, -2).contiguous()
    o, ht, notes = impl(x, inp.cu, run)
    if ht is not None and run.state_v_first:
        ht = ht.transpose(-1, -2)
    return _finish(lv, o, ht, inp, notes, backward)


# ---------------------------------------------------------------------------
# Torch-side emulation of options an implementation cannot do natively
# ---------------------------------------------------------------------------

def eda_gate32(x, A_log, dt_bias, l=LOWER_BOUND):
    """mixers.py ``safe_gate`` (EDA Denk. 15) in fp32."""
    A = A_log.float().exp()[:, None]
    return l + (-l) * torch.exp(-(A / abs(l)) * F.softplus(x.float() + dt_bias.float()))


def epilogue32(o, w, gate, act):
    """Per-head RMSNorm * weight * act(gate) in fp32 (fla FusedRMSNormGated semantics)."""
    o = o.float()
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    return y * (F.silu(gate.float()) if act == "silu" else torch.sigmoid(gate.float()))


# ---------------------------------------------------------------------------
# Implementation adapters: __call__(x, cu, run) -> (o, final_state, notes)
# ---------------------------------------------------------------------------

class ImplUnavailable(Exception):
    pass


class FlaImpl:
    """fla.ops.kda.chunk_kda. Native: precomputed / kimi / gdn gates, l2norm, h0 / final state, cu_seqlens,
    GVA, state_v_first, sigmoid beta (scale 1 or 2, if this FLA version has ``use_beta_sigmoid_in_kernel``).
    Emulated in torch: eda gate (precomputed in fp32), output epilogue (no epilogue in chunk_kda)."""
    name = "fla"
    supports_bwd = True

    def __init__(self):
        try:
            from fla.ops.kda import chunk_kda
        except ImportError as e:
            raise ImplUnavailable(f"fla not importable: {e}")
        self.fn = chunk_kda
        p = inspect.signature(chunk_kda).parameters
        self.native_beta = "use_beta_sigmoid_in_kernel" in p and "allow_neg_eigval" in p
        self.native_v_first = "state_v_first" in p

    def __call__(self, x, cu, run):
        notes, kw = {}, {}
        g, beta = x["g"], x["beta"]
        if run.gate_mode == "precomputed":
            notes["gate"] = "native"
        elif run.gate_mode in ("kimi", "gdn"):
            kw.update(use_gate_in_kernel=True, A_log=x["A_log"], dt_bias=x["dt_bias"].reshape(-1))
            if run.gate_mode == "kimi":
                kw["lower_bound"] = run.lower_bound
                kw["safe_gate"] = run.safe_gate
            notes["gate"] = "native"
        elif run.gate_mode == "eda":
            g, notes["gate"] = eda_gate32(g, x["A_log"], x["dt_bias"], run.lower_bound), "emulated"
        if run.beta_mode == "sigmoid":
            if self.native_beta and run.beta_scale in (1.0, 2.0):
                kw.update(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=run.beta_scale == 2.0)
                notes["beta"] = "native"
            else:
                beta, notes["beta"] = run.beta_scale * torch.sigmoid(beta.float()), "emulated"
        else:
            notes["beta"] = "native"
        if run.state_v_first:
            if not self.native_v_first:
                raise ImplUnavailable("this FLA has no state_v_first")
            kw["state_v_first"] = True
        o, ht = self.fn(x["q"], x["k"], x["v"], g, beta, initial_state=x.get("h0"),
                        output_final_state=run.final_state, use_qk_l2norm_in_kernel=run.l2,
                        cu_seqlens=cu, **kw)
        if run.epi:
            o, notes["epilogue"] = epilogue32(o, x["ep_w"], x["ep_gate"], run.epi), "emulated"
        return o, ht, notes


class QlaImpl:
    """Our unified entry point ``chunk_delta_rule``; every option is passed to it (gate forms, beta sigmoid, L2 norm and
    the epilogue run inside the kernels)."""
    name = "qla"
    supports_bwd = True
    CANDIDATES = (("mogan_kda", "chunk_delta_rule"),
                  ("mogan_kda.ops.kimi_delta_rule.chunk", "chunk_delta_rule"),
                  ("mogan_kda.ops.kimi_delta_rule.chunk", "chunk_kda"))

    def __init__(self):
        from arch_utils import ARCH
        if ARCH != "SM90":
            raise ImplUnavailable(f"qla KDA kernels target Hopper (SM90), found {ARCH}")
        import importlib
        for mod, attr in self.CANDIDATES:
            try:
                self.fn = getattr(importlib.import_module(mod), attr)
                return
            except (ImportError, AttributeError):
                continue
        raise ImplUnavailable("mogan_kda KDA entry point not available yet")

    def __call__(self, x, cu, run):
        on = dict(weight=x["ep_w"], gate=x["ep_gate"], act=run.epi, eps=1e-6) if run.epi else None
        o, ht = self.fn(
            x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x.get("h0"),
            output_final_state=run.final_state, cu_seqlens=cu, state_v_first=run.state_v_first,
            gate_mode=run.gate_mode, A_log=x.get("A_log"), dt_bias=x.get("dt_bias"),
            lower_bound=run.lower_bound, beta_mode=run.beta_mode, beta_scale=run.beta_scale,
            use_qk_l2norm_in_kernel=run.l2, output_norm=on)
        return o, ht, {"kernels": "cumsum+kkt_solve_kda+fused_fwd_kda", "options": "in-kernel (gate, beta, l2, epilogue)"}


@functools.lru_cache(maxsize=None)
def load_impl(name):
    return {"fla": FlaImpl, "qla": QlaImpl}[name]()


# ---------------------------------------------------------------------------
# Session report (JSON written by a module fixture in test_kda_unit.py when KDA_REPORT is set)
# ---------------------------------------------------------------------------

RECORDS = []


def record(**kw):
    RECORDS.append(kw)
    print("KDA-METRIC " + json.dumps(kw, default=str))


def write_report():
    path = os.environ.get("KDA_REPORT")
    if path:
        with open(path, "w") as f:
            json.dump({"thresholds": THRESH, "records": RECORDS}, f, indent=1, default=str)
