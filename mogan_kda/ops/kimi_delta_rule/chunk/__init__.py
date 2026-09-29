# Per-channel-gate (KDA) chunk kernels: forward chain, backward chain and the API entry points (chunk_delta_rule, chunk_kda).
import math
import os

import torch
import torch.nn.functional as F

from mogan_kda.ops.utils import chunk_local_cumsum, group_reduce_vector
from mogan_kda.utils import l2norm_fwd, l2norm_bwd

from ..gate import kda_gate_cumsum_fwd, kda_gate_bwd
from ..epilogue import rmsnorm_gated, rmsnorm_gated_fwd, rmsnorm_gated_bwd

from .hopper import (
    kkt_solve_kda, fused_fwd_kda, scaled_operands_kda, bwd_w_kda, bwd_dh_kda, bwd_post_kda, bwd_intra_kda,
)

CHUNK_SIZE = 64
RCP_LN2 = 1.0 / math.log(2.0)
L2_EPS = 1e-6


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
    output_h: bool = False,
    check_lower_bound: bool = False,
    g_c: torch.Tensor | None = None,
):
    """KDA forward. q, k [B, T, HK, 128] bf16; v [B, T, HV, 128] bf16; g [B, T, HV, 128] fp32 PER-TOKEN natural-log
    decay (contract: >= -5); beta [B, T, HV] fp32; initial_state fp32 [N, HV, K, V] ([N, HV, V, K] if state_v_first).
    Returns ``(o, ht, g_c, A, Aqk, h)``: g_c = chunk-local log2 cumsum of g, A = Akk^-1, Aqk (bf16 [B, T, HV, 64]),
    h = per-chunk entering states (bf16, only if output_h). If ``g_c`` is given (fused gate + cumsum kernel,
    ``kda_gate_cumsum_fwd``) ``g`` may be None and the cumsum is skipped. The gate-scaled operands are not returned (the
    backward recomputes them from q, k, g_c)."""
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert beta.dtype == torch.float32
    assert q.shape[-1] == v.shape[-1] == 128
    if g_c is None:
        assert g.dtype == torch.float32 and g.shape[-1] == 128
    assert v.shape[2] % q.shape[2] == 0, "GVA needs HV % HK == 0"
    for t in (q, k, v, g if g_c is None else g_c, beta):
        assert t.is_contiguous()
    if initial_state is not None:
        assert initial_state.dtype == torch.float32
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "cu_seqlens requires B == 1"
    scale = q.shape[-1] ** -0.5 if scale is None else scale

    if g_c is None:
        g_c = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, scale=RCP_LN2)
    A, Aqk, qt, kt, kb, gl = kkt_solve_kda(
        q, k, g_c, beta, scale=scale, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens,
        check_lower_bound=check_lower_bound, emit_scaled=True,
    )
    o, h, ht = fused_fwd_kda(
        qt, kt, kb, v, A, Aqk, gl, beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, output_h=output_h, cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
    )
    return o, ht, g_c, A, Aqk, h


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_c: torch.Tensor,
    A: torch.Tensor,
    Aqk: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
    return_dh0: bool | None = None,
    return_dg_prime: bool = False,
):
    """KDA backward from the forward's saved tensors (q, k, v, beta, g_c, A, Aqk, h; see ``chunk_kda_fwd``).

    Chain: K0 scaled operands -> K1 w (negated) -> K2 reverse state chain
    (dh, dVd, dh0) -> K3a / K3b post chain (dv, dbeta_w, dAqk, dL, dq_p, dk_p, dg_p) -> K4 intra-chunk terms (dq, dk,
    dg', dbeta_g) -> K5: reverse chunk-local cumsum of dg' (natural-log per-token gradient: the ln2 factors cancel, no
    scale), GVA sum of dq / dk, dbeta = dbeta_w + dbeta_g. Deterministic (no atomics).

    ``return_dg_prime=True`` skips K5's reverse cumsum and returns K4's dg' as ``dg`` (``kda_gate_bwd(...,
    dg_is_prime=True)`` folds the cumsum into the gate backward).

    Returns ``(dq [B,T,HK,K] bf16, dk bf16, dv [B,T,HV,V] bf16, dg [B,T,HV,K] fp32, dbeta [B,T,HV] fp32, dh0)``;
    dh0 fp32 [N,HV,K,V] ([N,HV,V,K] with state_v_first) when ``return_dh0`` (default: ``initial_state is not None``)."""
    K = q.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    return_dh0 = (initial_state is not None) if return_dh0 is None else return_dh0
    do = do.to(torch.bfloat16).contiguous()
    if dht is not None:
        dht = dht.float().contiguous()
    HK, HV = q.shape[2], v.shape[2]

    qt, kt, kb, gl = scaled_operands_kda(q, k, g_c, cu_seqlens)
    nw = bwd_w_kda(A, beta, kt, cu_seqlens, negate=True)
    dh, dvd, dh0 = bwd_dh_kda(do, Aqk, nw, qt, kb, gl, dht, scale, cu_seqlens, state_v_first, output_dh0=return_dh0)
    post = bwd_post_kda(do, v, h, dh, dvd, A, beta, qt, kt, kb, gl, g_c, scale, cu_seqlens, state_v_first,
                       q=q, k=k)
    del qt, kt, kb, nw, dh, dvd
    dq, dk, dg_prime, dbeta_g = bwd_intra_kda(
        q, k, g_c, beta, post["dAqk"], post["dL"], post["dq_p"], post["dk_p"], post["dg_p"], scale, cu_seqlens,
    )
    # K5
    if return_dg_prime:
        dg = dg_prime
    else:
        dg = chunk_local_cumsum(dg_prime, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, reverse=True,
                                zero_last_chunk=True)
    if HV > HK:
        dq, dk = group_reduce_vector(dq, HK), group_reduce_vector(dk, HK)  # deterministic GVA group sum
    dbeta = post["dbeta_w"] + dbeta_g
    return dq.to(q.dtype), dk.to(k.dtype), post["dv"], dg, dbeta, dh0


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, state_v_first):
        need_bwd = any(ctx.needs_input_grad[i] for i in (0, 1, 2, 3, 4, 6))
        o, ht, g_c, A, Aqk, h = chunk_kda_fwd(
            q, k, v, g, beta, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens, state_v_first=state_v_first,
            output_h=need_bwd,
        )
        if need_bwd:
            ctx.save_for_backward(q, k, v, beta, g_c, A, Aqk, h, initial_state, cu_seqlens)
        ctx.scale = q.shape[-1] ** -0.5 if scale is None else scale
        ctx.state_v_first = state_v_first
        ctx.need_dh0 = ctx.needs_input_grad[6]
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, beta, g_c, A, Aqk, h, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, dg, dbeta, dh0 = chunk_kda_bwd(
            q, k, v, beta, g_c, A, Aqk, h, do, dht, initial_state, ctx.scale, cu_seqlens, ctx.state_v_first,
            return_dh0=ctx.need_dh0,
        )
        dh0 = dh0.float() if (initial_state is not None and ctx.need_dh0) else None
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.float(), dbeta.float(), None, dh0,
                None, None, None)


# ---------------------------------------------------------------------------
# Torch reference of the options (gate forms, beta, L2, epilogue); used by the scalar (GDN) path and the tests
# ---------------------------------------------------------------------------

def _log_decay(g, mode, A_log, dt_bias, lower_bound):
    """Natural-log decay from the raw pre-activation ``g`` ([B, T, HV] scalar or [B, T, HV, K] per-channel).
    Modes: "precomputed" -> g; "gdn" -> -exp(A_log) softplus(g + dt_bias) (clamped to >= lower_bound if not None);
    "kimi" -> lower_bound sigmoid(exp(A_log) (g + dt_bias)); "eda" -> l + (-l) exp(-(exp(A_log) / |l|) softplus(g + dt_bias)),
    l = lower_bound. A_log [HV]; dt_bias [HV] (scalar g) or [HV, K] / flat [HV * K] (per-channel g)."""
    x = g.float()
    if mode == "precomputed":
        return x.contiguous()
    if dt_bias is not None:
        x = x + dt_bias.float().reshape(x.shape[2:])
    A = A_log.float().exp().reshape(x.shape[2], *([1] * (x.dim() - 3))) if A_log is not None else None
    if mode == "gdn":
        out = -A * F.softplus(x)
        return (out if lower_bound is None else out.clamp(min=lower_bound)).contiguous()
    if mode == "kimi":
        return (lower_bound * torch.sigmoid(x if A is None else A * x)).contiguous()
    if mode == "eda":
        return (lower_bound + (-lower_bound) * torch.exp(-(A / abs(lower_bound)) * F.softplus(x))).contiguous()
    raise ValueError(f"unknown gate_mode {mode!r}")


def _beta(beta, mode, beta_scale):
    if mode == "precomputed":
        return beta.float().contiguous()
    if mode == "sigmoid":
        return (beta_scale * torch.sigmoid(beta.float())).contiguous()
    raise ValueError(f"unknown beta_mode {mode!r}")


def _l2norm(x):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + L2_EPS)).to(x.dtype)


def _output_norm(o, weight, gate, act="silu", eps=1e-6):
    """Per-head RMSNorm(o) * weight * act(gate), fp32 (fla FusedRMSNormGated semantics)."""
    o = o.float()
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    gate = gate.float()
    return y * (F.silu(gate) if act in ("silu", "swish") else torch.sigmoid(gate))


class ChunkDeltaRuleKDAFunction(torch.autograd.Function):
    """Whole KDA layer with in-kernel options. forward: gate + chunk cumsum (kda_gate_cumsum_fwd) -> beta sigmoid ->
    q/k L2 norm -> chunk_kda_fwd(g_c) -> output epilogue. backward: epilogue bwd -> chunk_kda_bwd (dg', no K5) ->
    kda_gate_bwd (K5 folded) -> beta / L2 backward. Grads: q, k, v, x_gate, beta_raw, A_log, dt_bias, h0, ep_w, ep_gate."""

    @staticmethod
    def forward(ctx, q, k, v, x, beta, A_log, dt_bias, initial_state, ep_w, ep_gate, cfg):
        need_bwd = any(ctx.needs_input_grad[:10])
        g_c = kda_gate_cumsum_fwd(x, cfg["gate_mode"], A_log, dt_bias, cfg["lower_bound"], cfg["cu_seqlens"])
        beta_e = _beta(beta, cfg["beta_mode"], cfg["beta_scale"])
        q_rstd = k_rstd = None
        qn, kn = q, k
        if cfg["l2"]:
            qn, q_rstd = l2norm_fwd(q)
            kn, k_rstd = l2norm_fwd(k)
        o, ht, g_c, A, Aqk, h = chunk_kda_fwd(
            qn, kn, v, None, beta_e, scale=cfg["scale"], initial_state=initial_state,
            output_final_state=cfg["output_final_state"], cu_seqlens=cfg["cu_seqlens"],
            state_v_first=cfg["state_v_first"], output_h=need_bwd, g_c=g_c,
        )
        y = rmsnorm_gated_fwd(o, ep_w, ep_gate, cfg["ep_act"], cfg["ep_eps"]) if cfg["has_ep"] else o
        if need_bwd:
            ctx.save_for_backward(qn, kn, q_rstd, k_rstd, v, beta, x, A_log, dt_bias, g_c, A, Aqk, h, initial_state,
                                  ep_w, ep_gate, o if cfg["has_ep"] else None)
        ctx.cfg = cfg
        ctx.scale = q.shape[-1] ** -0.5 if cfg["scale"] is None else cfg["scale"]
        ctx.need_dh0 = ctx.needs_input_grad[7]
        return y, ht

    @staticmethod
    def backward(ctx, dy, dht):
        cfg = ctx.cfg
        (qn, kn, q_rstd, k_rstd, v, beta, x, A_log, dt_bias, g_c, A, Aqk, h, initial_state, ep_w, ep_gate,
         o) = ctx.saved_tensors
        cu = cfg["cu_seqlens"]
        dw = dgate = None
        if cfg["has_ep"]:
            do, dw, dgate = rmsnorm_gated_bwd(dy, o, ep_w, ep_gate, cfg["ep_act"], cfg["ep_eps"])
        else:
            do = dy
        beta_e = _beta(beta, cfg["beta_mode"], cfg["beta_scale"])
        dq, dk, dv, dg, dbeta, dh0 = chunk_kda_bwd(
            qn, kn, v, beta_e, g_c, A, Aqk, h, do, dht, initial_state, ctx.scale, cu, cfg["state_v_first"],
            return_dh0=ctx.need_dh0, return_dg_prime=True,
        )
        dx, dA, ddt = kda_gate_bwd(
            x, dg.float().contiguous(), cfg["gate_mode"], A_log, dt_bias, cfg["lower_bound"], cu,
            dg_is_prime=True, zero_tail=cu is not None,
        )
        if cfg["beta_mode"] == "sigmoid":
            sg = torch.sigmoid(beta.float())
            dbeta = dbeta * (cfg["beta_scale"] * sg * (1.0 - sg))
        if cfg["l2"]:
            dq = l2norm_bwd(qn, q_rstd, dq)
            dk = l2norm_bwd(kn, k_rstd, dk)
        dh0 = dh0.float() if (initial_state is not None and ctx.need_dh0) else None
        return (dq.to(qn.dtype), dk.to(kn.dtype), dv.to(v.dtype), dx, dbeta.to(beta.dtype), dA, ddt, dh0, dw, dgate,
                None)


def _chunk_delta_rule_emulated(
    q, k, v, g, beta, *, scale, initial_state, output_final_state, cu_seqlens, state_v_first,
    gate_mode, A_log, dt_bias, lower_bound, beta_mode, beta_scale, use_qk_l2norm_in_kernel, output_norm,
    auto_cp, cp_context,
):
    """Torch emulation of all options around the kernels (``MOGAN_KDA_EMULATE=1``; used by the tests as a reference)."""
    g = _log_decay(g, gate_mode, A_log, dt_bias, lower_bound)
    beta = _beta(beta, beta_mode, beta_scale)
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    if g.dim() == 3:
        from mogan_kda.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

        o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            state_v_first=state_v_first, auto_cp=auto_cp, cp_context=cp_context,
        )
    else:
        o, ht = ChunkKDAFunction.apply(
            q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, state_v_first,
        )
    if output_norm is not None:
        o = _output_norm(o, output_norm.get("weight"), output_norm["gate"],
                         output_norm.get("act", "silu"), output_norm.get("eps", 1e-6))
    return o, ht


def chunk_delta_rule(
    q, k, v, g, beta, *,
    scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, state_v_first=False,
    gate_mode="precomputed", A_log=None, dt_bias=None, lower_bound=-5.0,
    beta_mode="precomputed", beta_scale=1.0,
    use_qk_l2norm_in_kernel=False, output_norm=None,
    auto_cp=True, cp_context=None,
):
    """Single entry point of the unified delta rule. g [B, T, HV] -> scalar gate (GDN kernels),
    g [B, T, HV, K] -> per-channel gate (KDA kernels). Returns ``(o, final_state)``.

    KDA path (4-D g): everything runs in kernels inside ``ChunkDeltaRuleKDAFunction`` (fused gate activation +
    chunk cumsum, beta sigmoid (tiny torch elementwise on [B, T, HV]), q/k L2 norm (``l2norm_fwd/bwd``),
    output epilogue kernel, gate backward with folded K5). ``gate_mode="gdn"`` ENFORCES ``lower_bound`` (None ->
    -5.0, the contract of the KDA kernels). ``output_norm=dict(weight, gate, act, eps)``; the result is then fp32.
    Scalar path (3-D g): gate and beta stay in torch (autograd), L2 norm is native in ``chunk_gated_delta_rule``,
    the epilogue is our kernel. ``auto_cp`` / ``cp_context`` are honoured only by the scalar path (no CP for KDA).
    ``MOGAN_KDA_EMULATE=1`` runs the options in torch around the kernels (reference for the tests)."""
    if os.environ.get("MOGAN_KDA_EMULATE", "0") == "1":
        return _chunk_delta_rule_emulated(
            q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, state_v_first=state_v_first, gate_mode=gate_mode, A_log=A_log, dt_bias=dt_bias,
            lower_bound=lower_bound, beta_mode=beta_mode, beta_scale=beta_scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, output_norm=output_norm, auto_cp=auto_cp,
            cp_context=cp_context,
        )
    ep = output_norm
    if g.dim() == 3:
        from mogan_kda.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

        o, ht = chunk_gated_delta_rule(
            q.contiguous(), k.contiguous(), v.contiguous(), _log_decay(g, gate_mode, A_log, dt_bias, lower_bound),
            _beta(beta, beta_mode, beta_scale), scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens, state_v_first=state_v_first, auto_cp=auto_cp, cp_context=cp_context,
        )
        if ep is not None:
            o = rmsnorm_gated(o.contiguous(), ep.get("weight"), ep["gate"], ep.get("act", "silu"),
                              ep.get("eps", 1e-6))
        return o, ht

    if gate_mode == "gdn" and lower_bound is None:
        lower_bound = -5.0
    cfg = dict(
        scale=scale, output_final_state=output_final_state, cu_seqlens=cu_seqlens, state_v_first=state_v_first,
        gate_mode=gate_mode, lower_bound=lower_bound, beta_mode=beta_mode, beta_scale=beta_scale,
        l2=use_qk_l2norm_in_kernel, has_ep=ep is not None,
        ep_act=None if ep is None else ep.get("act", "silu"), ep_eps=None if ep is None else ep.get("eps", 1e-6),
    )
    return ChunkDeltaRuleKDAFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), A_log, dt_bias,
        initial_state, None if ep is None else ep.get("weight"), None if ep is None else ep["gate"].contiguous(), cfg,
    )


def chunk_kda(q, k, v, g, beta, **kwargs):
    """Kimi Delta Attention (per-channel gate): ``chunk_delta_rule`` restricted to 4-D ``g`` [B, T, HV, K]."""
    assert g.dim() == 4, "chunk_kda needs a per-channel gate g [B, T, HV, K]; use chunk_gated_delta_rule for scalar gates"
    return chunk_delta_rule(q, k, v, g, beta, **kwargs)


__all__ = ["chunk_kda", "chunk_kda_fwd", "chunk_kda_bwd", "ChunkKDAFunction", "ChunkDeltaRuleKDAFunction", "chunk_delta_rule", "kkt_solve_kda", "fused_fwd_kda"]
