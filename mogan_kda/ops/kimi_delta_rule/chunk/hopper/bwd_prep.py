# Derived from mogan_kda/ops/gated_delta_rule/chunk/hopper/kkt_solve.py
# (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT License).
"""K0 and K1 of the KDA backward (simple 128-thread, chunk x v-head kernels).

K0 ``scaled_operands_kda`` regenerates the gate-scaled operands of the forward from q, k and the chunk-local log2
cumsum g (the same tensors ``kkt_solve_kda(emit_scaled=True)`` writes; a standalone kernel because re-running the
kkt kernel would recompute A / Aqk for nothing)::

    q~ = q 2^g   k~ = k 2^g   kbar_j = k_j 2^(g_L - g_j)   gl = 2^(g_L)      (g_L = g of the last VALID row of the chunk)

    qt, kt, kb  bf16 [B, T, HV, K] (per v-head, rows past a sequence end are NOT written),
    gl          fp32 [lead, NC, HV, K], chunk id laid out like FlashQLA's ``h``: dense [B, ceil(T/64)], varlen [1, total_chunks].
    Every exponent argument is <= 0 (g is non-increasing inside a chunk); no clamp is needed.

K1 ``bwd_w_kda``: w = Akk diag(beta) k~ (bf16 [B, T, HV, K]); ``negate=True`` returns -w, which K2 consumes
(-w^T dVd is accumulated with a plain GEMM). beta * k~ is rounded to bf16 once, like the forward's operands.

Both match ``torch_ref.scaled_operands_torch`` / ``bwd_w_torch``. Access is contiguous along the channel axis and all
conditions are row-uniform. Shared memory: K0 ~1 KB (elementwise straight from global), K1 56 KB.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from mogan_kda.utils import prepare_chunk_indices

_POLICY = GemmWarpPolicy.FullRow


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_scaled_operands_kda(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qk_dtype,
    seqlen_dtype,
    is_varlen,
    single_batch,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    gl_batch = T.dynamic("gl_batch")
    gl_chunks = T.dynamic("gl_chunks")
    block_S = chunk_size

    q_shape = (data_batch_size, num_tokens, Hg, DK)
    g_shape = (data_batch_size, num_tokens, H, DK)
    gl_shape = (gl_batch, gl_chunks, H, DK)

    @T.macro
    def kernel_body(bb, bc, bh, bhg, chunk_idx, seq_start_idx, seq_end_idx, q, k, g, qt, kt, kb, gl):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        gL_shared = T.alloc_shared((DK,), dtype=accum_dtype, scope="shared")
        last_row = T.if_then_else(right <= seq_end_idx, block_S - 1, seq_end_idx - left - 1)

        for j_k in T.Parallel(DK):
            gL_shared[j_k] = g[bb, left + last_row, bh, j_k]
        for j_k in T.Parallel(DK):
            gl[bb, bc if is_varlen else chunk_idx, bh, j_k] = T.exp2(gL_shared[j_k])
        for j_s, j_k in T.Parallel(block_S, DK):
            if left + j_s < seq_end_idx:
                gv = g[bb, left + j_s, bh, j_k]
                e = T.exp2(gv)
                qt[bb, left + j_s, bh, j_k] = q[bb, left + j_s, bhg, j_k] * e
                kt[bb, left + j_s, bh, j_k] = k[bb, left + j_s, bhg, j_k] * e
                kb[bb, left + j_s, bh, j_k] = k[bb, left + j_s, bhg, j_k] * T.exp2(gL_shared[j_k] - gv)

    if is_varlen:

        @T.prim_func
        def tilelang_scaled_operands_kda_kernel(
            q: T.Tensor(q_shape, dtype=qk_dtype),
            k: T.Tensor(q_shape, dtype=qk_dtype),
            g: T.Tensor(g_shape, dtype=accum_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            qt: T.Tensor(g_shape, dtype=qk_dtype),
            kt: T.Tensor(g_shape, dtype=qk_dtype),
            kb: T.Tensor(g_shape, dtype=qk_dtype),
            gl: T.Tensor(gl_shape, dtype=accum_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(bb, bc, bh, bhg, chunk_idx, seq_start_idx, seq_end_idx, q, k, g, qt, kt, kb, gl)

    else:

        @T.prim_func
        def tilelang_scaled_operands_kda_kernel(
            q: T.Tensor(q_shape, dtype=qk_dtype),
            k: T.Tensor(q_shape, dtype=qk_dtype),
            g: T.Tensor(g_shape, dtype=accum_dtype),
            qt: T.Tensor(g_shape, dtype=qk_dtype),
            kt: T.Tensor(g_shape, dtype=qk_dtype),
            kb: T.Tensor(g_shape, dtype=qk_dtype),
            gl: T.Tensor(gl_shape, dtype=accum_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                if single_batch:
                    kernel_body(0, bc, bh, bhg, bc, 0, num_tokens, q, k, g, qt, kt, kb, gl)
                else:
                    bb = T.alloc_var("int32")
                    chunk_idx = T.alloc_var("int32")
                    seq_start_idx = T.alloc_var("int32")
                    seq_end_idx = T.alloc_var("int32")

                    # int32 locals evaluated once (a plain expression would be re-inlined into every global address)
                    bb = bc % data_batch_size
                    chunk_idx = bc // data_batch_size
                    seq_start_idx = 0
                    seq_end_idx = num_tokens

                    kernel_body(bb, bc, bh, bhg, chunk_idx, seq_start_idx, seq_end_idx, q, k, g, qt, kt, kb, gl)

    return tilelang_scaled_operands_kda_kernel


def scaled_operands_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    """K0. q, k [B, T, HK, 128] bf16; g [B, T, HV, 128] fp32 chunk-local log2 cumsum. Returns (qt, kt, kb, gl)."""
    B, Tn, Hg, K = q.shape
    H = g.shape[2]
    assert K == 128 and chunk_size == 64 and H % Hg == 0
    assert q.dtype == k.dtype == torch.bfloat16 and g.dtype == torch.float32
    assert q.is_contiguous() and k.is_contiguous() and g.is_contiguous() and g.shape == (B, Tn, H, K)
    if cu_seqlens is None:
        num_chunks = B * tilelang.cdiv(Tn, chunk_size)
        seqlen_dtype, is_varlen = "int32", False
        gl = torch.empty((B, tilelang.cdiv(Tn, chunk_size), H, K), dtype=torch.float32, device=q.device)
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype, is_varlen = cu_seqlens.dtype, True
        gl = torch.empty((1, len(chunk_indices), H, K), dtype=torch.float32, device=q.device)
    qt, kt, kb = (torch.empty((B, Tn, H, K), dtype=q.dtype, device=q.device) for _ in range(3))
    kernel = tilelang_scaled_operands_kda(
        H, Hg, K, chunk_size, accum_dtype="float32", qk_dtype=q.dtype, seqlen_dtype=seqlen_dtype, is_varlen=is_varlen,
        single_batch=(not is_varlen and B == 1),
    )
    if is_varlen:
        kernel(q, k, g, cu_seqlens, chunk_indices, qt, kt, kb, gl)
    else:
        kernel(q, k, g, qt, kt, kb, gl, num_chunks)
    return qt, kt, kb, gl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_bwd_w_kda(
    H,
    DK,
    chunk_size,
    accum_dtype,
    qk_dtype,
    seqlen_dtype,
    is_varlen,
    single_batch,
    negate,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    k_shape = (data_batch_size, num_tokens, H, DK)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(bb, bh, chunk_idx, seq_start_idx, seq_end_idx, a, beta, kt, w):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        a_shared = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
        kt_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
        kbeta_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
        b_shared = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")
        w_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
        w_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)

        if right <= seq_end_idx:
            T.async_copy(a[bb, left:right, bh, 0:block_S], a_shared)
            T.async_copy(kt[bb, left:right, bh, 0:DK], kt_shared)
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = beta[bb, left + j_s, bh]
            T.ptx_wait_group(0)
        else:  # zero padded rows: Akk, k~ and beta of rows past the end are exactly zero
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[bb, left + j_s, bh, j_t]
                else:
                    a_shared[j_s, j_t] = 0
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    kt_shared[j_s, j_k] = kt[bb, left + j_s, bh, j_k]
                else:
                    kt_shared[j_s, j_k] = 0
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = beta[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0

        # beta * k~ (bf16 GEMM operand)
        for j_s, j_k in T.Parallel(block_S, DK):
            kbeta_shared[j_s, j_k] = kt_shared[j_s, j_k] * b_shared[j_s]
        T.fence_proxy_async()
        # w = Akk @ (beta k~)
        T.gemm(a_shared, kbeta_shared, w_fragment, clear_accum=True, policy=_POLICY)
        if negate:
            for j_s, j_k in T.Parallel(block_S, DK):
                w_fragment[j_s, j_k] = w_fragment[j_s, j_k] * -1.0
        T.copy(w_fragment, w_shared)
        if right <= seq_end_idx:
            T.copy(w_shared, w[bb, left:right, bh, 0:DK])
        else:
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    w[bb, left + j_s, bh, j_k] = w_shared[j_s, j_k]

    if is_varlen:

        @T.prim_func
        def tilelang_bwd_w_kda_kernel(
            a: T.Tensor(a_shape, dtype=qk_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            kt: T.Tensor(k_shape, dtype=qk_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            w: T.Tensor(k_shape, dtype=qk_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(bb, bh, chunk_idx, seq_start_idx, seq_end_idx, a, beta, kt, w)

    else:

        @T.prim_func
        def tilelang_bwd_w_kda_kernel(
            a: T.Tensor(a_shape, dtype=qk_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            kt: T.Tensor(k_shape, dtype=qk_dtype),
            w: T.Tensor(k_shape, dtype=qk_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                if single_batch:
                    kernel_body(0, bh, bc, 0, num_tokens, a, beta, kt, w)
                else:
                    bb = T.alloc_var("int32")
                    chunk_idx = T.alloc_var("int32")
                    seq_start_idx = T.alloc_var("int32")
                    seq_end_idx = T.alloc_var("int32")

                    bb = bc % data_batch_size
                    chunk_idx = bc // data_batch_size
                    seq_start_idx = 0
                    seq_end_idx = num_tokens

                    kernel_body(bb, bh, chunk_idx, seq_start_idx, seq_end_idx, a, beta, kt, w)

    return tilelang_bwd_w_kda_kernel


def bwd_w_kda(
    A: torch.Tensor,
    beta: torch.Tensor,
    kt: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    negate: bool = False,
    chunk_size: int = 64,
):
    """K1. A [B, T, HV, 64] bf16 (Akk^-1), beta [B, T, HV] fp32, kt [B, T, HV, 128] bf16 (from K0).
    Returns w = Akk diag(beta) k~ (bf16 [B, T, HV, 128]), or -w when ``negate``."""
    B, Tn, H, K = kt.shape
    assert K == 128 and chunk_size == 64
    assert A.dtype == kt.dtype == torch.bfloat16 and beta.dtype == torch.float32
    assert A.shape == (B, Tn, H, chunk_size) and beta.shape == (B, Tn, H)
    assert A.is_contiguous() and beta.is_contiguous() and kt.is_contiguous()
    if cu_seqlens is None:
        num_chunks = B * tilelang.cdiv(Tn, chunk_size)
        seqlen_dtype, is_varlen = "int32", False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype, is_varlen = cu_seqlens.dtype, True
    w = torch.empty_like(kt)
    kernel = tilelang_bwd_w_kda(
        H, K, chunk_size, accum_dtype="float32", qk_dtype=kt.dtype, seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen, negate=negate, single_batch=(not is_varlen and B == 1),
    )
    if is_varlen:
        kernel(A, beta, kt, cu_seqlens, chunk_indices, w)
    else:
        kernel(A, beta, kt, w, num_chunks)
    return w
