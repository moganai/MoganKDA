"""fp64 naive (token-by-token) reference for the unified delta-rule API: scalar gate (GDN) and
per-channel gate (KDA), GVA, varlen, initial/final state, in-reference gate / beta / output epilogue.

Pure PyTorch, float64 regardless of input dtype, device-agnostic (runs on CPU), differentiable
(no in-place ops on the autograd path), so it can be used for gradchecks on tiny shapes.
All returned tensors are float64 (NOT cast back to the input dtype).

Recurrence (per v-head h, state S in R^{Dk x Dv}; q/k head = h // (HV // HK)), FLA
``naive_recurrent_kda`` (fla/ops/kda/naive.py) and the GDN recurrence of FlashQLA::

    S <- diag(exp(g_t)) S                   # decay along the K axis (scalar gate: same value on every row)
    S <- S + beta_t k_t (v_t - S^T k_t)^T   # delta rule, i.e. (I - beta k k^T) diag(a) S + beta k v^T
    o_t = (scale q_t)^T S                   # output after the update, scale defaults to Dk ** -0.5

Conventions:
  * ``g`` in mode "precomputed" is the PER-TOKEN log-decay (natural log, NOT cumsum'ed). The chunked
    kernels cumsum it internally (naive_chunk_kda does ``g.cumsum``).
  * State layout is [N, HV, Dk, Dv]; with ``state_v_first`` it is [N, HV, Dv, Dk] (initial and final).
    N = B without cu_seqlens, else number of segments.
  * ``cu_seqlens`` requires B == 1; each segment starts from its own initial_state row (or zeros).
  * ``use_qk_l2norm_in_kernel``: x / sqrt(sum(x^2) + 1e-6) over the last dim (fla ``l2norm_fwd`` and
    ``tests/ref_gdr.py::torch_l2norm_fwd``), applied to q/k at HK heads before GVA expansion.

Gate modes (``g`` is the raw pre-activation ``x``; x' = x + dt_bias, A = exp(A_log)):
  * "precomputed": log-decay = g
  * "gdn"  : -A * softplus(x')                        (fla kda gate w/o lower_bound; GDN/Mamba form).
             If ``lower_bound`` is not None the result is additionally clamped to >= lower_bound
             (assumption for the plan's "GDN softplus + lower bound"; pass lower_bound=None for the pure form).
  * "kimi" : lower_bound * sigmoid(A * x')            (fla ``naive_kda_lowerbound_gate``; A_log/dt_bias optional)
  * "eda"  : l + (-l) * exp(-(A/|l|) * softplus(x')), l = lower_bound   (mixers.py ``safe_gate``, Denk. 15)
  Shapes (as mixers.py / model.py): A_log [HV] always; scalar gate g [B,T,HV] with dt_bias [HV];
  per-channel g [B,T,HV,Dk] with dt_bias [HV,Dk] (FLA's flat [HV*Dk] is accepted). A_log broadcasts over Dk.
"""

import torch
import torch.nn.functional as F

L2_EPS = 1e-6
_F64 = torch.float64


# ---------------------------------------------------------------------------
# Helpers reused by tests
# ---------------------------------------------------------------------------

def ref_gate(g, mode="precomputed", A_log=None, dt_bias=None, lower_bound=-5.0):
    """Log-decay from the raw pre-activation ``g`` ([B,T,HV] or [B,T,HV,Dk]); see module docstring."""
    x = g.to(_F64)
    if mode == "precomputed":
        return x
    if dt_bias is not None:
        x = x + dt_bias.to(_F64).reshape(x.shape[2:])
    A = A_log.to(_F64).exp().reshape(x.shape[2], *([1] * (x.dim() - 3))) if A_log is not None else None
    if mode == "gdn":
        assert A is not None, "gate_mode='gdn' needs A_log"
        out = -A * F.softplus(x)
        return out if lower_bound is None else out.clamp(min=lower_bound)
    if mode == "kimi":
        assert lower_bound is not None, "gate_mode='kimi' needs lower_bound"
        return lower_bound * torch.sigmoid(x if A is None else A * x)
    if mode == "eda":
        assert A is not None and lower_bound is not None, "gate_mode='eda' needs A_log and lower_bound"
        return lower_bound + (-lower_bound) * torch.exp(-(A / abs(lower_bound)) * F.softplus(x))
    raise ValueError(f"unknown gate_mode {mode!r}")


def ref_beta(beta, mode="precomputed", scale=1.0):
    """beta [B,T,HV]: "precomputed" -> beta, "sigmoid" -> scale * sigmoid(beta) (model.py: beta_olcek * sigmoid)."""
    b = beta.to(_F64)
    if mode == "precomputed":
        return b
    if mode == "sigmoid":
        return scale * torch.sigmoid(b)
    raise ValueError(f"unknown beta_mode {mode!r}")


def ref_output_norm(o, weight, gate, act="silu", eps=1e-6):
    """Per-head gated RMSNorm exactly as fla ``FusedRMSNormGated`` (used in model.py ``o_norm(o, g)``):
        y = o / sqrt(mean(o^2, -1) + eps) * weight;   y = y * act(gate)
    o: [B,T,HV,Dv]; weight: [Dv] (shared over heads) or None; gate: [B,T,HV,Dv] (model.py ``kapi(x)`` viewed per head);
    act: "silu" (= swish) -> gate*sigmoid(gate) | "sigmoid"."""
    o = o.to(_F64)
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.to(_F64)
    gate = gate.to(_F64)
    if act in ("silu", "swish"):
        return y * gate * torch.sigmoid(gate)
    if act == "sigmoid":
        return y * torch.sigmoid(gate)
    raise ValueError(f"unknown act {act!r}")


def _l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------

def _recurrent(q, k, v, g, beta, S):
    """q,k,g pre-expanded to HV heads ([B,T,HV,Dk]; q already scaled), v [B,T,HV,Dv], beta [B,T,HV],
    S [B,HV,Dk,Dv]. Returns (o [B,T,HV,Dv], S)."""
    outs = []
    for t in range(q.shape[1]):
        k_t = k[:, t]
        S = S * torch.exp(g[:, t])[..., None]
        pred = (k_t[..., None] * S).sum(-2)                                  # S^T k_t  [B,HV,Dv]
        S = S + (beta[:, t][..., None] * k_t)[..., None] * (v[:, t] - pred)[..., None, :]
        outs.append((q[:, t][..., None] * S).sum(-2))                        # (scale q_t)^T S
    return torch.stack(outs, dim=1), S


def ref_chunk_delta_rule(
    q, k, v, g, beta, *,
    scale=None, initial_state=None, output_final_state=False,
    cu_seqlens=None, state_v_first=False,
    gate_mode="precomputed", A_log=None, dt_bias=None, lower_bound=-5.0,
    beta_mode="precomputed", beta_scale=1.0,
    use_qk_l2norm_in_kernel=False, output_norm=None,
):
    """Reference for the unified delta-rule API. q,k [B,T,HK,Dk]; v [B,T,HV,Dv]; g [B,T,HV] (scalar gate)
    or [B,T,HV,Dk] (per-channel); beta [B,T,HV]. Returns (o [B,T,HV,Dv] float64, final_state or None).

    ``output_norm``: None | dict(weight=[Dv] or None, gate=[B,T,HV,Dv], act="silu"|"sigmoid", eps=1e-6);
    when given, ``o`` is the epilogue output ``ref_output_norm(o, ...)``.
    """
    B, T, HK, Dk = q.shape
    HV, Dv = v.shape[2], v.shape[3]
    assert k.shape == q.shape and v.shape[:2] == (B, T)
    assert HV % HK == 0, "GVA needs HV % HK == 0"
    assert g.shape[:3] == (B, T, HV) and g.dim() in (3, 4), "g must be [B,T,HV] or [B,T,HV,Dk]"
    assert g.dim() == 3 or g.shape[3] == Dk
    assert beta.shape == (B, T, HV)
    G = HV // HK
    scale = Dk ** -0.5 if scale is None else scale

    q, k, v = q.to(_F64), k.to(_F64), v.to(_F64)
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q = q.repeat_interleave(G, dim=2) * scale        # v-head h uses q/k head h // G
    k = k.repeat_interleave(G, dim=2)
    g = ref_gate(g, gate_mode, A_log, dt_bias, lower_bound)
    if g.dim() == 3:                                 # scalar gate: same decay on every K row
        g = g[..., None].expand(B, T, HV, Dk)
    beta = ref_beta(beta, beta_mode, beta_scale)

    if cu_seqlens is None:
        bounds, N = [(0, T)], B                      # batch dim vectorized, state per batch row
    else:
        assert B == 1, "cu_seqlens requires B == 1"
        cu = [int(x) for x in cu_seqlens.tolist()]
        assert cu[0] == 0 and cu[-1] == T
        bounds = list(zip(cu[:-1], cu[1:]))
        N = len(bounds)
    if initial_state is not None:
        assert initial_state.shape[0] == N
        h0 = initial_state.to(_F64)
        if state_v_first:
            h0 = h0.transpose(-1, -2)                # -> [N, HV, Dk, Dv]

    o_parts, states = [], []
    for n, (s, e) in enumerate(bounds):
        if initial_state is None:
            S = q.new_zeros(B, HV, Dk, Dv)
        else:
            S = h0 if cu_seqlens is None else h0[n:n + 1]
        o_n, S = _recurrent(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], S)
        o_parts.append(o_n)
        states.append(S)
    o = torch.cat(o_parts, dim=1)

    if output_norm is not None:
        o = ref_output_norm(o, output_norm.get("weight"), output_norm["gate"],
                            output_norm.get("act", "silu"), output_norm.get("eps", 1e-6))
    final_state = None
    if output_final_state:
        final_state = torch.cat(states, dim=0)
        if state_v_first:
            final_state = final_state.transpose(-1, -2).contiguous()
    return o, final_state


# ---------------------------------------------------------------------------
# FLA chunk_kda-compatible wrapper
# ---------------------------------------------------------------------------

def ref_kda_fla_compat(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, A_log=None, dt_bias=None,
    safe_gate=False, lower_bound=None, cu_seqlens=None,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, **unsupported,
):
    """Same call convention as ``fla.ops.kda.chunk_kda`` (g [B,T,HV,Dk]) mapped onto ``ref_chunk_delta_rule``.

    Mapping: use_gate_in_kernel=False -> "precomputed". True -> "kimi" if lower_bound is not None (FLA's
    kernel switches to ``lower_bound * sigmoid(exp(A_log) (g + dt_bias))`` whenever lower_bound is set,
    A_log optional), else FLA's ``-exp(A_log) softplus(g + dt_bias)`` = mode "gdn" with lower_bound=None.
    use_beta_sigmoid_in_kernel -> beta_mode "sigmoid" with scale 2 if allow_neg_eigval else 1.
    ``safe_gate`` only selects the FLA kernel path (M=16 TensorCore) and does not change the math.
    Not mappable (raise if set to a non-default value): cp_context, use_graph, max_num_seqs, cu_seqlens_cpu,
    disable_recompute, return_intermediate_states, chunk_size (numerically irrelevant for a recurrent ref).
    """
    for name, val in unsupported.items():
        if val not in (None, False, 64):
            raise NotImplementedError(f"chunk_kda argument {name!r} is not supported by the reference")
    mode = "precomputed"
    if use_gate_in_kernel:
        mode = "kimi" if lower_bound is not None else "gdn"
    return ref_chunk_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, state_v_first=state_v_first,
        gate_mode=mode, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
        beta_mode="sigmoid" if use_beta_sigmoid_in_kernel else "precomputed",
        beta_scale=2.0 if allow_neg_eigval else 1.0,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
