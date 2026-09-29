"""Fused per-channel gate activation + chunk-local cumsum for KDA (forward and backward), TileLang.

Forward  ``kda_gate_cumsum_fwd(x, mode, A_log, dt_bias, lower_bound, cu_seqlens) -> g_c``
    x [B, T, HV, K] raw pre-activation (fp32 or bf16) -> g_c fp32 [B, T, HV, K]: chunk-local (64 tokens, per sequence)
    cumsum of the natural-log decay, times 1/ln2 (log2 domain). Same layout / semantics as
    ``chunk_local_cumsum(_log_decay(x, ...), scale=1/ln2)``.
Backward ``kda_gate_bwd(x, dg, mode, A_log, dt_bias, lower_bound, cu_seqlens, dg_is_prime) -> (dx, dA_log, ddt_bias)``
    dg = gradient w.r.t. the natural-log per-token decay (what ``chunk_kda_bwd`` returns). With ``dg_is_prime=True`` the
    input is instead K4's ``dg'`` (before K5) and the chunk-local REVERSE cumsum (K5) is folded into this kernel.

Gate modes (s = x + dt_bias, E = exp(A_log), l = lower_bound < 0):
    "eda"        g = l + (-l) exp(-(E / |l|) softplus(s))
    "kimi"       g = l sigmoid(E s)
    "gdn"        g = max(-E softplus(s), l)     the lower bound is ENFORCED (contract of the KDA kernels: g >= -5);
                                                the gradient is 0 where the clamp is active (g0 < l)
    "precomputed" g = x                         (plain cumsum; A_log / dt_bias unused)

Kernel design (both kernels): grid = num_chunks * NB programs, 128 threads; one program = one (64-token chunk,
32-channel block) tile (channels c = h * K + k are viewed as NB blocks of BH = 32; K % 32 == 0, a block never straddles
a head). smem: one fp32 [64, 33] transpose tile (padded against bank conflicts). Same tile scheme as
``chunk_local_cumsum`` (transpose through smem, ``T.cumsum`` along the token axis of the [BH, 64] fragment).
Deterministic reductions (no atomics): each backward program reduces its tile over the 64 tokens (``T.reduce_sum``,
fixed order) into a per-(chunk, channel) partial; ddt_bias / dA_log are then the torch sums of the partials over the
chunk axis (and over k within a head for dA_log), a second, fixed-shape (hence deterministic) reduction.
"""
import math

import torch
import tilelang
import tilelang.language as T

from mogan_kda.utils import prepare_chunk_indices
from mogan_kda.ops.utils import chunk_local_cumsum

CHUNK_SIZE = 64
BLOCK_C = 32  # channels per program
RCP_LN2 = 1.0 / math.log(2.0)
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
MODES = ("eda", "kimi", "gdn", "precomputed")


# ---------------------------------------------------------------------------
# elementwise math (module-level helpers returning expressions; only T.min / T.max, no per-element selects)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    """1 / (1 + exp(-z)); the exp2 argument is capped so that it never overflows to inf."""
    return 1.0 / (1.0 + T.exp2(T.min(-z * _LOG2E, 100.0)))


def _softplus(s):
    """max(s, 0) + log(1 + exp(-|s|)) (stable, == torch softplus up to ~1e-7 absolute)."""
    return T.max(s, 0.0) + T.log2(1.0 + T.exp2(-T.max(s, -s) * _LOG2E)) * _LN2


def _gate_value(mode, s, E, lb, abs_lb):
    if mode == "gdn":
        return T.max(-E * _softplus(s), lb)
    if mode == "kimi":
        return lb * _sigmoid(E * s)
    if mode == "eda":
        return lb + (-lb) * T.exp2(-(E / abs_lb) * _softplus(s) * _LOG2E)
    return s  # precomputed


def _gate_grad(mode, which, s, E, G, lb, abs_lb):
    """dL/ds (which == 0) or dL/dA_log (which == 1) of one element for the incoming natural-log gradient G."""
    if mode == "gdn":
        sp = _softplus(s)
        # clamp mask (torch clamp semantics: gradient passes where -E sp >= l), arithmetic 0/1 select
        keep = T.min(T.max((-E * sp - lb) * 1e30 + 1.0, 0.0), 1.0)
        return G * keep * (-E) * (_sigmoid(s) if which == 0 else sp)
    if mode == "kimi":
        u = E * s
        sg = _sigmoid(u)
        return G * lb * sg * (1.0 - sg) * (E if which == 0 else u)
    # eda
    sp = _softplus(s)
    c = E / abs_lb
    w = T.exp2(-c * sp * _LOG2E)
    return G * lb * w * c * (_sigmoid(s) if which == 0 else sp)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

@tilelang.jit
def tilelang_kda_gate_cumsum(
    NB, BH, K, HV, mode, lower_bound, x_dtype, seqlen_dtype, is_varlen, single_batch, scale, chunk_size=CHUNK_SIZE,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    blocks_per_head = K // BH
    abs_lb = abs(lower_bound)

    x_shape = (data_batch_size, num_tokens, NB, BH)

    @T.macro
    def kernel_body(bb, cb, chunk_idx, seq_start_idx, seq_end_idx, x, bias, A_log, g_c):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        x_frag = T.alloc_fragment((block_S, BH), dtype=x_dtype)
        s_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        g_fragment = T.alloc_fragment((BH, block_S), dtype="float32")
        o_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        gT_shared = T.alloc_shared((block_S, BH + 1), dtype="float32")

        if right <= seq_end_idx:
            T.copy(x[bb, left:right, cb, :], x_frag)
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    x_frag[j, i] = x[bb, left + j, cb, i]
                else:
                    x_frag[j, i] = 0

        # s = x + dt_bias (fp32)
        for j, i in T.Parallel(block_S, BH):
            s_frag[j, i] = T.Cast("float32", x_frag[j, i]) + bias[cb * BH + i]  # bias = 0 for "precomputed"

        # natural-log gate -> transpose tile
        for j, i in T.Parallel(block_S, BH):
            gT_shared[j, i] = _gate_value(
                mode, s_frag[j, i], T.exp2(A_log[cb // blocks_per_head] * _LOG2E), lower_bound, abs_lb,
            )

        for i, j in T.Parallel(BH, block_S):
            g_fragment[i, j] = gT_shared[j, i]

        T.cumsum(g_fragment, dim=1)

        for i, j in T.Parallel(BH, block_S):
            g_fragment[i, j] = g_fragment[i, j] * scale

        for i, j in T.Parallel(BH, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :BH], o_frag)
        if right <= seq_end_idx:
            T.copy(o_frag, g_c[bb, left:right, cb, :])
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    g_c[bb, left + j, cb, i] = o_frag[j, i]

    if is_varlen:

        @T.prim_func
        def kda_gate_cumsum_kernel(
            x: T.Tensor(x_shape, dtype=x_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            bias: T.Tensor([NB * BH], dtype="float32"),
            A_log: T.Tensor([HV], dtype="float32"),
            g_c: T.Tensor(x_shape, dtype="float32"),
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]
                kernel_body(0, cb, chunk_idx, seq_start_idx, seq_end_idx, x, bias, A_log, g_c)

    else:

        @T.prim_func
        def kda_gate_cumsum_kernel(
            x: T.Tensor(x_shape, dtype=x_dtype),
            bias: T.Tensor([NB * BH], dtype="float32"),
            A_log: T.Tensor([HV], dtype="float32"),
            g_c: T.Tensor(x_shape, dtype="float32"),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                if single_batch:  # B == 1: constant batch index (provable alignment -> vectorised accesses)
                    kernel_body(0, cb, bc, 0, num_tokens, x, bias, A_log, g_c)
                else:
                    bb = T.alloc_var("int32")
                    chunk_idx = T.alloc_var("int32")
                    seq_start_idx = T.alloc_var("int32")
                    seq_end_idx = T.alloc_var("int32")

                    # int32 locals evaluated once (a plain expression is re-inlined into every global address)
                    bb = bc % data_batch_size
                    chunk_idx = bc // data_batch_size
                    seq_start_idx = 0
                    seq_end_idx = num_tokens

                    kernel_body(bb, cb, chunk_idx, seq_start_idx, seq_end_idx, x, bias, A_log, g_c)

    return kda_gate_cumsum_kernel


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------

@tilelang.jit
def tilelang_kda_gate_bwd(
    NB, BH, K, HV, mode, lower_bound, x_dtype, seqlen_dtype, is_varlen, single_batch, fold_reverse_cumsum,
    chunk_size=CHUNK_SIZE,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_programs = T.dynamic("num_programs")
    block_S = chunk_size
    blocks_per_head = K // BH
    abs_lb = abs(lower_bound)

    x_shape = (data_batch_size, num_tokens, NB, BH)

    @T.macro
    def kernel_body(bb, bcn, cb, chunk_idx, seq_start_idx, seq_end_idx, x, dg, bias, A_log, dx, part_b, part_a):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        x_frag = T.alloc_fragment((block_S, BH), dtype=x_dtype)
        dg_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        s_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        dz_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        da_frag = T.alloc_fragment((block_S, BH), dtype="float32")
        dx_frag = T.alloc_fragment((block_S, BH), dtype=x_dtype)
        g_fragment = T.alloc_fragment((BH, block_S), dtype="float32")
        db_red = T.alloc_fragment((BH,), dtype="float32")
        da_red = T.alloc_fragment((BH,), dtype="float32")
        dg_shared = T.alloc_shared((block_S, BH + 1), dtype="float32")

        if right <= seq_end_idx:
            T.copy(x[bb, left:right, cb, :], x_frag)
            T.copy(dg[bb, left:right, cb, :], dg_frag)
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    x_frag[j, i] = x[bb, left + j, cb, i]
                    dg_frag[j, i] = dg[bb, left + j, cb, i]
                else:
                    x_frag[j, i] = 0
                    dg_frag[j, i] = 0

        T.copy(dg_frag, dg_shared[:, :BH])
        if fold_reverse_cumsum:  # K5: dg = reverse chunk-local cumsum of dg' (padded rows are 0 at the tail)
            for i, j in T.Parallel(BH, block_S):
                g_fragment[i, j] = dg_shared[j, i]
            T.cumsum(g_fragment, dim=1, reverse=True)
            for i, j in T.Parallel(BH, block_S):
                dg_shared[j, i] = g_fragment[i, j]

        for j, i in T.Parallel(block_S, BH):
            s_frag[j, i] = T.Cast("float32", x_frag[j, i]) + bias[cb * BH + i]

        for j, i in T.Parallel(block_S, BH):
            dz_frag[j, i] = _gate_grad(
                mode, 0, s_frag[j, i], T.exp2(A_log[cb // blocks_per_head] * _LOG2E), dg_shared[j, i],
                lower_bound, abs_lb,
            )
            da_frag[j, i] = _gate_grad(
                mode, 1, s_frag[j, i], T.exp2(A_log[cb // blocks_per_head] * _LOG2E), dg_shared[j, i],
                lower_bound, abs_lb,
            )

        for j, i in T.Parallel(block_S, BH):
            dx_frag[j, i] = dz_frag[j, i]

        # per-(chunk, channel) partials over the 64 tokens (padded rows contribute exactly 0: dg = 0 there)
        T.reduce_sum(dz_frag, db_red, dim=0, clear=True)
        T.reduce_sum(da_frag, da_red, dim=0, clear=True)
        for i in T.Parallel(BH):
            part_b[bcn, i] = db_red[i]
            part_a[bcn, i] = da_red[i]

        if right <= seq_end_idx:
            T.copy(dx_frag, dx[bb, left:right, cb, :])
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    dx[bb, left + j, cb, i] = dx_frag[j, i]

    if is_varlen:

        @T.prim_func
        def kda_gate_bwd_kernel(
            x: T.Tensor(x_shape, dtype=x_dtype),
            dg: T.Tensor(x_shape, dtype="float32"),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            bias: T.Tensor([NB * BH], dtype="float32"),
            A_log: T.Tensor([HV], dtype="float32"),
            dx: T.Tensor(x_shape, dtype=x_dtype),
            part_b: T.Tensor([num_programs, BH], dtype="float32"),
            part_a: T.Tensor([num_programs, BH], dtype="float32"),
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]
                kernel_body(0, bcn, cb, chunk_idx, seq_start_idx, seq_end_idx, x, dg, bias, A_log, dx, part_b, part_a)

    else:

        @T.prim_func
        def kda_gate_bwd_kernel(
            x: T.Tensor(x_shape, dtype=x_dtype),
            dg: T.Tensor(x_shape, dtype="float32"),
            bias: T.Tensor([NB * BH], dtype="float32"),
            A_log: T.Tensor([HV], dtype="float32"),
            dx: T.Tensor(x_shape, dtype=x_dtype),
            part_b: T.Tensor([num_programs, BH], dtype="float32"),
            part_a: T.Tensor([num_programs, BH], dtype="float32"),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                if single_batch:
                    kernel_body(0, bcn, cb, bc, 0, num_tokens, x, dg, bias, A_log, dx, part_b, part_a)
                else:
                    bb = T.alloc_var("int32")
                    chunk_idx = T.alloc_var("int32")
                    seq_start_idx = T.alloc_var("int32")
                    seq_end_idx = T.alloc_var("int32")

                    bb = bc % data_batch_size
                    chunk_idx = bc // data_batch_size
                    seq_start_idx = 0
                    seq_end_idx = num_tokens

                    kernel_body(
                        bb, bcn, cb, chunk_idx, seq_start_idx, seq_end_idx, x, dg, bias, A_log, dx, part_b, part_a,
                    )

    return kda_gate_bwd_kernel


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------

def _check_args(x, mode, A_log, lower_bound):
    assert mode in MODES, f"unknown gate mode {mode!r}"
    assert x.dim() == 4 and x.is_contiguous(), "x must be a contiguous [B, T, HV, K] tensor"
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.shape[-1] % BLOCK_C == 0, f"K={x.shape[-1]} must be a multiple of {BLOCK_C}"
    if mode != "precomputed":
        assert lower_bound is not None and lower_bound < 0, f"gate mode {mode!r} needs lower_bound < 0"
        assert mode == "kimi" or A_log is not None, f"gate mode {mode!r} needs A_log"


def _params(x, A_log, dt_bias):
    """fp32 A_log [HV] (zeros = exp(0) = 1 if absent) and flat fp32 dt_bias [HV*K] (zeros if absent)."""
    HV, K = x.shape[2], x.shape[3]
    A = torch.zeros(HV, device=x.device, dtype=torch.float32) if A_log is None else A_log.float().contiguous()
    bias = torch.zeros(HV * K, device=x.device, dtype=torch.float32) if dt_bias is None \
        else dt_bias.float().reshape(HV * K).contiguous()
    return A, bias


def _launch_geometry(x, cu_seqlens, chunk_size=CHUNK_SIZE):
    B, Tn, HV, K = x.shape
    NB = HV * K // BLOCK_C
    if cu_seqlens is None:
        return NB, B * tilelang.cdiv(Tn, chunk_size), "int32", None
    assert B == 1, "cu_seqlens requires B == 1"
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return NB, len(chunk_indices), cu_seqlens.dtype, chunk_indices


def kda_gate_cumsum_fwd(
    x: torch.Tensor,
    mode: str,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = RCP_LN2,
) -> torch.Tensor:
    """Fused gate activation + chunk-local cumsum (see module docstring). x [B, T, HV, K] fp32 / bf16 (contiguous),
    A_log [HV], dt_bias [HV, K] or flat [HV * K]; returns fp32 g_c [B, T, HV, K] = cumsum_chunk(gate(x)) * scale
    (default 1/ln2). Tokens past ``cu_seqlens[-1]`` are left uninitialised (as ``chunk_local_cumsum``)."""
    _check_args(x, mode, A_log, lower_bound)
    B, Tn, HV, K = x.shape
    NB, num_chunks, seqlen_dtype, chunk_indices = _launch_geometry(x, cu_seqlens)
    precomputed = mode == "precomputed"
    A, bias = _params(x, None if precomputed else A_log, None if precomputed else dt_bias)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_kda_gate_cumsum(
        NB, BLOCK_C, K, HV, mode, float(lower_bound if lower_bound is not None else -1.0), x.dtype, seqlen_dtype,
        is_varlen, (not is_varlen and B == 1), float(scale),
    )
    g_c = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    x_v, g_v = x.view(B, Tn, NB, BLOCK_C), g_c.view(B, Tn, NB, BLOCK_C)
    if is_varlen:
        kernel(x_v, cu_seqlens, chunk_indices, bias, A, g_v)
    else:
        kernel(x_v, bias, A, g_v, num_chunks)
    return g_c


def kda_gate_bwd(
    x: torch.Tensor,
    dg: torch.Tensor,
    mode: str,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    dg_is_prime: bool = False,
    zero_tail: bool = False,
):
    """Gate backward. ``dg`` fp32 [B, T, HV, K]: gradient w.r.t. the natural-log per-token gate (``chunk_kda_bwd``'s
    dg), or, with ``dg_is_prime=True``, K4's dg' with K5's chunk-local reverse cumsum folded in.
    Returns ``(dx [x.dtype], dA_log fp32 [HV] | None, ddt_bias fp32 (shape of dt_bias) | None)``; dA_log / ddt_bias are
    None for "precomputed" or when the parameter was not given. Deterministic (per-chunk partials + fixed-shape torch
    sums). ``zero_tail``: zero the tokens past ``cu_seqlens[-1]`` in dx (else undefined)."""
    _check_args(x, mode, A_log, lower_bound)
    assert dg.shape == x.shape and dg.dtype == torch.float32 and dg.is_contiguous()
    B, Tn, HV, K = x.shape
    if mode == "precomputed":
        if dg_is_prime:
            dx = chunk_local_cumsum(dg, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, reverse=True,
                                    zero_last_chunk=zero_tail)
        else:
            dx = dg
        return dx.to(x.dtype), None, None

    NB, num_chunks, seqlen_dtype, chunk_indices = _launch_geometry(x, cu_seqlens)
    A, bias = _params(x, A_log, dt_bias)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_kda_gate_bwd(
        NB, BLOCK_C, K, HV, mode, float(lower_bound), x.dtype, seqlen_dtype, is_varlen,
        (not is_varlen and B == 1), bool(dg_is_prime),
    )
    dx = torch.zeros_like(x) if zero_tail else torch.empty_like(x)
    part_b = torch.empty(num_chunks * NB, BLOCK_C, device=x.device, dtype=torch.float32)
    part_a = torch.empty_like(part_b)
    x_v, dg_v, dx_v = (t.view(B, Tn, NB, BLOCK_C) for t in (x, dg, dx))
    if is_varlen:
        kernel(x_v, dg_v, cu_seqlens, chunk_indices, bias, A, dx_v, part_b, part_a)
    else:
        kernel(x_v, dg_v, bias, A, dx_v, part_b, part_a, num_chunks)

    # second (fixed-shape, deterministic) reduction of the per-chunk partials
    ddt_bias = None
    if dt_bias is not None:
        ddt_bias = part_b.view(num_chunks, HV * K).sum(0).view(dt_bias.shape)
    dA_log = None
    if A_log is not None:
        dA_log = part_a.view(num_chunks, HV, K).sum(2).sum(0).view(A_log.shape)
    return dx, dA_log, ddt_bias


__all__ = ["kda_gate_cumsum_fwd", "kda_gate_bwd", "MODES"]
