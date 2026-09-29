# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from mogan_kda.utils import prepare_chunk_indices


@tilelang.jit(
    # out_idx=[-1],
)
def tilelang_chunk_local_cumsum(
    H,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
    zero_last_chunk,
    apply_scale=False,
    scale=1.0,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        g_fragment = T.alloc_fragment((H, block_S), dtype=accum_dtype)
        gT_fragment = T.alloc_fragment((block_S, H), dtype=g_dtype)
        gT_shared = T.alloc_shared((block_S, H + 1), dtype=g_dtype)

        if right <= seq_end_idx:
            T.copy(g_raw[bb, left:right, 0:H], gT_fragment)
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    gT_fragment[j, i] = g_raw[bb, left + j, i]
                else:
                    gT_fragment[j, i] = 0
        T.copy(gT_fragment, gT_shared[:, :H])

        for i, j in T.Parallel(H, block_S):
            g_fragment[i, j] = gT_shared[j, i]

        T.cumsum(g_fragment, dim=1, reverse=reverse)

        if apply_scale:
            for i, j in T.Parallel(H, block_S):
                g_fragment[i, j] = g_fragment[i, j] * scale

        for i, j in T.Parallel(H, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :H], gT_fragment)
        if right <= seq_end_idx:
            T.copy(gT_fragment, g_cumsum[bb, left:right, 0:H])
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    g_cumsum[bb, left + j, i] = gT_fragment[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

                if zero_last_chunk:
                    left = seq_start_idx + chunk_idx * block_S
                    if batch_idx == real_batch_size - 1 and left + block_S >= seq_end_idx:
                        for j, i in T.Parallel(block_S, H):
                            if seq_end_idx + j < num_tokens:
                                g_cumsum[bb, seq_end_idx + j, i] = 0

    else:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    return tilelang_chunk_local_cumsum_kernel


@tilelang.jit(
    # out_idx=[-1],
)
def tilelang_chunk_local_cumsum_vector(
    NB,
    BH,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
    zero_last_chunk,
    apply_scale=False,
    scale=1.0,
    single_batch=False,
):
    """Per-channel chunk-local cumsum. The channel axis (C = H * K of ``g [B, T, H, K]``) is viewed as
    ``NB`` blocks of ``BH`` channels: g_raw is ``[B, T, NB, BH]`` and one program handles one
    (chunk, channel block) tile of ``(chunk_size, BH)``. Same tile scheme as the scalar kernel above,
    with the block index ``cb`` selecting the channel block."""
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, NB, BH)

    @T.macro
    def kernel_body(
        bb,
        cb,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        g_fragment = T.alloc_fragment((BH, block_S), dtype=accum_dtype)
        gT_fragment = T.alloc_fragment((block_S, BH), dtype=g_dtype)
        gT_shared = T.alloc_shared((block_S, BH + 1), dtype=g_dtype)

        if right <= seq_end_idx:
            T.copy(g_raw[bb, left:right, cb, :], gT_fragment)
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    gT_fragment[j, i] = g_raw[bb, left + j, cb, i]
                else:
                    gT_fragment[j, i] = 0
        T.copy(gT_fragment, gT_shared[:, :BH])

        for i, j in T.Parallel(BH, block_S):
            g_fragment[i, j] = gT_shared[j, i]

        T.cumsum(g_fragment, dim=1, reverse=reverse)

        if apply_scale:
            for i, j in T.Parallel(BH, block_S):
                g_fragment[i, j] = g_fragment[i, j] * scale

        for i, j in T.Parallel(BH, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :BH], gT_fragment)
        if right <= seq_end_idx:
            T.copy(gT_fragment, g_cumsum[bb, left:right, cb, :])
        else:
            for j, i in T.Parallel(block_S, BH):
                if left + j < seq_end_idx:
                    g_cumsum[bb, left + j, cb, i] = gT_fragment[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_vector_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    cb,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

                if zero_last_chunk:
                    left = seq_start_idx + chunk_idx * block_S
                    if batch_idx == real_batch_size - 1 and left + block_S >= seq_end_idx:
                        for j, i in T.Parallel(block_S, BH):
                            if seq_end_idx + j < num_tokens:
                                g_cumsum[bb, seq_end_idx + j, cb, i] = 0

    else:

        @T.prim_func
        def tilelang_chunk_local_cumsum_vector_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * NB, threads=128) as (bcn,):
                bc, cb = bcn // NB, bcn % NB
                if single_batch:  # B == 1: constant batch index (provable alignment -> vectorised accesses)
                    kernel_body(0, cb, bc, 0, num_tokens, g_raw, g_cumsum)
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

                    kernel_body(
                        bb,
                        cb,
                        chunk_idx,
                        seq_start_idx,
                        seq_end_idx,
                        g_raw,
                        g_cumsum,
                    )

    return tilelang_chunk_local_cumsum_vector_kernel


# Channels per program of the per-channel cumsum. K (=128 in the first release) must be a multiple.
VECTOR_BLOCK_C = 32


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    reverse: bool = False,
    zero_last_chunk: bool = False,
    scale: float | None = None,
):
    """Chunk-local (per-sequence, per-chunk) cumsum of ``g`` along the token axis.

    ``g`` is ``[B, T, H]`` (scalar gate) or contiguous ``[B, T, H, K]`` (per-channel gate; ``K`` must be a
    multiple of ``VECTOR_BLOCK_C``). Accumulation is fp32, the output has the dtype of ``g``.
    ``scale`` (None = no multiply) is applied to the cumsum in fp32 before the store, e.g. ``1 / ln 2`` to
    get a log2-domain gate.
    """
    assert g.dim() in (3, 4)
    assert g.stride(-1) == 1
    is_vector = g.dim() == 4
    if is_vector:
        assert g.is_contiguous(), "per-channel cumsum needs a contiguous [B, T, H, K] gate"
        BH = VECTOR_BLOCK_C
        assert g.shape[-1] % BH == 0, f"K={g.shape[-1]} must be a multiple of {BH}"
        NB = g.shape[2] * g.shape[3] // BH
        g_view = g.view(g.shape[0], g.shape[1], NB, BH)  # channel-block view, same memory
        batch_size, num_tokens = g.shape[:2]
    else:
        batch_size, num_tokens, H = g.shape
        g_view = g

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    # zero_last_chunk is used by backward gradients.  Start from zero as the
    # padding can be longer than the single tile required by the next TMA
    # consumer, and the valid chunks below overwrite their complete domain.
    g_cumsum = torch.zeros_like(g) if zero_last_chunk else torch.empty_like(g)

    scale_kwargs = dict(apply_scale=scale is not None, scale=1.0 if scale is None else float(scale))
    if is_vector:
        kernel = tilelang_chunk_local_cumsum_vector(
            NB,
            BH,
            chunk_size,
            g_dtype=g.dtype,
            seqlen_dtype=seqlen_dtype,
            accum_dtype="float32",
            is_varlen=is_varlen,
            reverse=reverse,
            zero_last_chunk=zero_last_chunk,
            **scale_kwargs,
            single_batch=(not is_varlen and batch_size == 1),
        )
        out_view = g_cumsum.view_as(g_view)
    else:
        kernel = tilelang_chunk_local_cumsum(
            H,
            chunk_size,
            g_dtype=g.dtype,
            seqlen_dtype=seqlen_dtype,
            accum_dtype="float32",
            is_varlen=is_varlen,
            reverse=reverse,
            zero_last_chunk=zero_last_chunk,
            **scale_kwargs,
        )
        out_view = g_cumsum
    if is_varlen:
        kernel(g_view, cu_seqlens, chunk_indices, out_view)
    else:
        kernel(g_view, out_view, num_chunks)

    return g_cumsum
