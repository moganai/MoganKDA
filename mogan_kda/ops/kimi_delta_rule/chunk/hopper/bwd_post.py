# Derived from mogan_kda/ops/gated_delta_rule/chunk/hopper/kkt_solve.py and kkt_solve_kda.py
# (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT License).
"""K3a and K3b of the KDA backward: the chunk-parallel "post chain" (``torch_ref.bwd_post_torch``).

Per 64-token chunk and v-head, with S = h_c (state entering the chunk) and dS' = dh_c (gradient leaving it, both from
the forward / K2), Akk = A, and the K0 / K2 tensors::

    K3a (V looped in two halves of 64)
      X = V - k~ S          W = beta X (rows)          Vd = Akk W
      dAqk = tril(dO Vd^T)                      (lower incl. diagonal; the gradient w.r.t. the STORED Aqk)
      dW = Akk^T dVd        dX = beta dW = dV    dbeta_w = rowsum_v(dW (.) X)
      dAkk = dVd W^T        dL = -StrictLower(Akk^T dAkk Akk^T)
    K3b (K looped in four blocks of 32 channels)
      dq~ = scale dO S^T    dk~ = -dX S^T    dkbar = Vd dS'^T
      dq_p = dq~ 2^g        dk_p = dk~ 2^g + dkbar 2^(g_L - g)       g = chunk-local log2 cumsum, g_L = last VALID row
      dg'_p = q~ dq~ + k~ dk~ - kbar dkbar
      dg'_p[g_L row] += sum_j kbar_j dkbar_j + gl (.) dgl,        dgl_k = sum_v S_kv dS'_kv     (state-carry term)

dgl (moved here from K2): K3b holds the h / dh tiles of a K block, so dgl for that block is one elementwise product and
a row sum over V inside the CTA (no partials over V halves, no atomics).

Outputs (the interface K4 ``bwd_intra_kda`` and K5 expect): dv bf16 [B,T,HV,V] (final gradient, = dX), dbeta_w fp32
[B,T,HV], dAqk / dL bf16 [B,T,HV,64] (zeros outside the lower / strictly lower triangle, rows past a sequence end not
written), Vd bf16 [B,T,HV,V] (intermediate for K3b, kept for tests), dq_p / dk_p / dg_p fp32 [B,T,HV,K] (per v-head, added
to by K4; rows past a sequence end not written).

Why two kernels: K3a contracts over K (X) and over V halves (dAqk, dAkk) and keeps four 64x64 fp32 accumulators (97 KB
smem); K3b needs full-V operand tiles (dO, dX, Vd: 48 KB) plus the h / dh blocks of a K block (16 KB) and the gate-scaled
q~ / k~ / kbar / g slices; fusing them would keep X / Vd / dX and both state tiles resident (> 130 KB). The price is one
bf16 round trip of Vd (write 64 MB + read 64 MB at [1,16384,16,128]); dX = dv is written anyway and re-read.

Numerics: the exp2 arguments of K3b are <= 0 (g non-increasing; the caps are identities for in-range values and only
protect zero padded rows); no anchoring is needed (no exp of differences over pairs). GEMM operands are bf16 with fp32
accumulation; dAkk and P = dAkk Akk^T are split hi/lo (two bf16 GEMMs each, fp32 accumulate) for the sandwich.
Precision (A_log / dt_bias gradients are cancelling sums over all tokens): W = beta X is a hi/lo pair (Vd = Akk W, dAkk =
dVd W^T), K3a also writes the bf16 residuals dv_lo (dX - dv) and vd_lo (Vd - Vd_hi), K3b uses dX and Vd as hi + lo pairs
(dk~, dkbar), and with q / k given the dg' terms use fp32 q~ = q 2^g, k~ = k 2^g, kbar = k 2^(gL-g) (no bf16 q~/k~/kbar).
smem: K3a ~105 KB (the W lo tile aliases v_shared), K3b ~104 KB with split + q/k (g stays in registers): 2 CTA/SM.

Prefix invariance (bitwise; loss on rows < P, so dO, dVd (from K2) and dh of chunks from the one containing P on are exact
zeros): every reduction that mixes rows uses exact zeros for rows >= P: dAqk = tril(dO Vd^T); dW_j = sum_{i >= j} Akk_ij dVd_i
(dVd_i = 0 for i >= P); dL_ij (i > j) only involves dAkk_ab with b <= j < i, i.e. only W rows < P; dkbar = Vd dS'^T = 0 and
dgl = 0 in the chunk containing P and later (dS' = 0), so the g_L (future row) dependence of kbar / gl multiplies zeros.
Partial chunks are zero padded when loaded; nothing else is read; no atomics.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from mogan_kda.utils import prepare_chunk_indices

_POLICY = GemmWarpPolicy.FullRow


def _cap(d, cap):
    return T.min(d, cap)  # T.min, not a select on per-element values (defeats auto-vectorisation, probe GH200)


# ---------------------------------------------------------------------------
# K3a
# ---------------------------------------------------------------------------

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_bwd_post_a(
    H,
    DK,
    DV,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    seqlen_dtype,
    is_varlen,
    single_batch,
    state_v_first,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    h_batch = T.dynamic("h_batch")
    h_chunks = T.dynamic("h_chunks")
    block_S = chunk_size
    HALF = 64  # V columns per half

    k_shape = (data_batch_size, num_tokens, H, DK)
    v_shape = (data_batch_size, num_tokens, H, DV)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    h_shape = (h_batch, h_chunks, H, DV, DK) if state_v_first else (h_batch, h_chunks, H, DK, DV)

    @T.macro
    def kernel_body(
        bb,
        hb,
        gid,
        bh,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        v,
        do,
        dvd,
        kt,
        A,
        beta,
        h,
        dv,
        dbeta_w,
        dAqk,
        dL,
        vd_out,
        dv_lo,
        vd_lo,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
        kt_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")
        v_shared = T.alloc_shared((block_S, HALF), dtype=qkva_dtype)
        do_shared = T.alloc_shared((block_S, HALF), dtype=qkva_dtype)
        dvd_shared = T.alloc_shared((block_S, HALF), dtype=qkva_dtype)
        h_shared = T.alloc_shared((HALF, DK) if state_v_first else (DK, HALF), dtype=qkva_dtype)
        w_shared = T.alloc_shared((block_S, HALF), dtype=qkva_dtype)
        vd_shared = T.alloc_shared((block_S, HALF), dtype=qkva_dtype)
        dakk_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
        p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
        lo_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)  # low parts of the hi/lo split (sandwich)

        x_frag = T.alloc_fragment((block_S, HALF), dtype=accum_dtype)
        t_frag = T.alloc_fragment((block_S, HALF), dtype=accum_dtype)
        daq_frag = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        dak_frag = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        red_frag = T.alloc_fragment((block_S,), dtype=accum_dtype)
        db_shared = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")  # dbeta_w accumulator over the V halves

        # ---- chunk-level inputs: Akk, k~, beta (zero padded past the sequence end) ----
        if right <= seq_end_idx:
            T.async_copy(A[bb, left:right, bh, 0:block_S], a_shared)
            T.async_copy(kt[bb, left:right, bh, 0:DK], kt_shared)
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = beta[bb, left + j_s, bh]
            T.ptx_wait_group(0)
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = A[bb, left + j_s, bh, j_t]
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

        T.clear(daq_frag)
        T.clear(dak_frag)
        for j_s in T.Parallel(block_S):
            db_shared[j_s] = 0

        for i_h in T.serial(DV // HALF):
            c0 = i_h * HALF
            # ---- half-V inputs ----
            if right <= seq_end_idx:
                T.async_copy(v[bb, left:right, bh, c0 : c0 + HALF], v_shared)
                T.async_copy(do[bb, left:right, bh, c0 : c0 + HALF], do_shared)
                T.async_copy(dvd[bb, left:right, bh, c0 : c0 + HALF], dvd_shared)
                if state_v_first:
                    T.async_copy(h[hb, gid, bh, c0 : c0 + HALF, 0:DK], h_shared)
                else:
                    T.async_copy(h[hb, gid, bh, 0:DK, c0 : c0 + HALF], h_shared)
                T.ptx_wait_group(0)
            else:
                for j_s, j_v in T.Parallel(block_S, HALF):
                    if left + j_s < seq_end_idx:
                        v_shared[j_s, j_v] = v[bb, left + j_s, bh, c0 + j_v]
                        do_shared[j_s, j_v] = do[bb, left + j_s, bh, c0 + j_v]
                        dvd_shared[j_s, j_v] = dvd[bb, left + j_s, bh, c0 + j_v]
                    else:
                        v_shared[j_s, j_v] = 0
                        do_shared[j_s, j_v] = 0
                        dvd_shared[j_s, j_v] = 0
                if state_v_first:
                    T.copy(h[hb, gid, bh, c0 : c0 + HALF, 0:DK], h_shared)
                else:
                    T.copy(h[hb, gid, bh, 0:DK, c0 : c0 + HALF], h_shared)
            T.fence_proxy_async()

            # X = V - k~ S ; W = beta X
            if state_v_first:
                T.gemm(kt_shared, h_shared, x_frag, transpose_B=True, clear_accum=True, policy=_POLICY)
            else:
                T.gemm(kt_shared, h_shared, x_frag, clear_accum=True, policy=_POLICY)
            for j_s, j_v in T.Parallel(block_S, HALF):
                x_frag[j_s, j_v] = v_shared[j_s, j_v] - x_frag[j_s, j_v]
            # W = hi + lo (bf16 each; v_shared is free after X and holds the low part): the rounding of W feeds Vd
            # (dAqk, dkbar) and dAkk (dL), all weighted sums that end in the cancelling A_log / dt_bias reductions
            for j_s, j_v in T.Parallel(block_S, HALF):
                xw = x_frag[j_s, j_v] * b_shared[j_s]
                w_shared[j_s, j_v] = xw
                v_shared[j_s, j_v] = xw - w_shared[j_s, j_v]
            T.fence_proxy_async()

            # Vd = Akk (W_hi + W_lo) (kept for K3b), dAqk += dO (Vd_hi + Vd_lo)^T (dO is exact bf16)
            T.gemm(a_shared, w_shared, t_frag, clear_accum=True, policy=_POLICY)
            T.gemm(a_shared, v_shared, t_frag, clear_accum=False, policy=_POLICY)
            # Vd stays fp32 in t_frag: hi = bf16(Vd) is stored / used by K3b, lo = bf16(Vd - hi) only feeds dAqk
            for j_s, j_v in T.Parallel(block_S, HALF):
                x = t_frag[j_s, j_v]
                vd_shared[j_s, j_v] = x
                lo_shared[j_s, j_v] = x - vd_shared[j_s, j_v]
            T.fence_proxy_async()
            # hi and lo parts of Vd both go to K3b (dkbar = Vd dS'^T)
            if right <= seq_end_idx:
                T.copy(vd_shared, vd_out[bb, left:right, bh, c0 : c0 + HALF])
                T.copy(lo_shared, vd_lo[bb, left:right, bh, c0 : c0 + HALF])
            else:
                for j_s, j_v in T.Parallel(block_S, HALF):
                    if left + j_s < seq_end_idx:
                        vd_out[bb, left + j_s, bh, c0 + j_v] = vd_shared[j_s, j_v]
                        vd_lo[bb, left + j_s, bh, c0 + j_v] = lo_shared[j_s, j_v]
            T.gemm(do_shared, vd_shared, daq_frag, transpose_B=True, clear_accum=False, policy=_POLICY)
            T.gemm(do_shared, lo_shared, daq_frag, transpose_B=True, clear_accum=False, policy=_POLICY)

            # dW = Akk^T dVd ; dAkk += dVd W^T
            T.gemm(a_shared, dvd_shared, t_frag, transpose_A=True, clear_accum=True, policy=_POLICY)
            T.gemm(dvd_shared, w_shared, dak_frag, transpose_B=True, clear_accum=False, policy=_POLICY)
            T.gemm(dvd_shared, v_shared, dak_frag, transpose_B=True, clear_accum=False, policy=_POLICY)

            # dbeta_w += rowsum_v(dW (.) X)
            for j_s, j_v in T.Parallel(block_S, HALF):
                x_frag[j_s, j_v] = x_frag[j_s, j_v] * t_frag[j_s, j_v]
            T.reduce_sum(x_frag, red_frag, dim=1, clear=True)
            for j_s in T.Parallel(block_S):
                db_shared[j_s] = db_shared[j_s] + red_frag[j_s]

            # dV = dX = beta dW (w_shared / v_shared are free again): hi = the final bf16 dv, lo = residual for K3b
            # (dk~ = -dX S^T)
            for j_s, j_v in T.Parallel(block_S, HALF):
                xd = t_frag[j_s, j_v] * b_shared[j_s]
                w_shared[j_s, j_v] = xd
                v_shared[j_s, j_v] = xd - w_shared[j_s, j_v]
            T.fence_proxy_async()
            if right <= seq_end_idx:
                T.copy(w_shared, dv[bb, left:right, bh, c0 : c0 + HALF])
                T.copy(v_shared, dv_lo[bb, left:right, bh, c0 : c0 + HALF])
            else:
                for j_s, j_v in T.Parallel(block_S, HALF):
                    if left + j_s < seq_end_idx:
                        dv[bb, left + j_s, bh, c0 + j_v] = w_shared[j_s, j_v]
                        dv_lo[bb, left + j_s, bh, c0 + j_v] = v_shared[j_s, j_v]

        # ---- dbeta_w ----
        for j_s in T.Parallel(block_S):
            if left + j_s < seq_end_idx:
                dbeta_w[bb, left + j_s, bh] = db_shared[j_s]

        # ---- dAqk = tril(dO Vd^T) (lower incl. diagonal), bf16 ----
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= j_t:
                p_shared[j_s, j_t] = daq_frag[j_s, j_t]
            else:
                p_shared[j_s, j_t] = 0
        # dAkk (fp32 accumulator) -> hi / lo bf16 tiles for the sandwich (2^-17 instead of 2^-9: the A_log / dt_bias
        # gradients are weighted sums of dL over all tokens and must not see the rounding of these intermediates)
        for j_s, j_t in T.Parallel(block_S, block_S):
            x = dak_frag[j_s, j_t]
            dakk_shared[j_s, j_t] = x
            lo_shared[j_s, j_t] = x - dakk_shared[j_s, j_t]
        T.fence_proxy_async()
        if right <= seq_end_idx:
            T.copy(p_shared, dAqk[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    dAqk[bb, left + j_s, bh, j_t] = p_shared[j_s, j_t]

        # ---- dL = -StrictLower(Akk^T dAkk Akk^T) = -StrictLower(Akk^T P), P = dAkk Akk^T; Akk is exact bf16, dAkk and P
        # are split hi + lo (two GEMMs each into one fp32 accumulator) ----
        T.gemm(dakk_shared, a_shared, dak_frag, transpose_B=True, clear_accum=True, policy=_POLICY)
        T.gemm(lo_shared, a_shared, dak_frag, transpose_B=True, clear_accum=False, policy=_POLICY)
        for j_s, j_t in T.Parallel(block_S, block_S):
            x = dak_frag[j_s, j_t]
            p_shared[j_s, j_t] = x
            lo_shared[j_s, j_t] = x - p_shared[j_s, j_t]
        T.fence_proxy_async()
        T.gemm(a_shared, p_shared, daq_frag, transpose_A=True, clear_accum=True, policy=_POLICY)
        T.gemm(a_shared, lo_shared, daq_frag, transpose_A=True, clear_accum=False, policy=_POLICY)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s > j_t:
                dakk_shared[j_s, j_t] = daq_frag[j_s, j_t] * -1.0
            else:
                dakk_shared[j_s, j_t] = 0
        T.fence_proxy_async()
        if right <= seq_end_idx:
            T.copy(dakk_shared, dL[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    dL[bb, left + j_s, bh, j_t] = dakk_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_bwd_post_a_kernel(
            v: T.Tensor(v_shape, dtype=qkva_dtype),
            do: T.Tensor(v_shape, dtype=qkva_dtype),
            dvd: T.Tensor(v_shape, dtype=qkva_dtype),
            kt: T.Tensor(k_shape, dtype=qkva_dtype),
            A: T.Tensor(a_shape, dtype=qkva_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            h: T.Tensor(h_shape, dtype=qkva_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            dbeta_w: T.Tensor(b_shape, dtype=accum_dtype),
            dAqk: T.Tensor(a_shape, dtype=qkva_dtype),
            dL: T.Tensor(a_shape, dtype=qkva_dtype),
            vd_out: T.Tensor(v_shape, dtype=qkva_dtype),
            dv_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            vd_lo: T.Tensor(v_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    0, 0, bc, bh, chunk_idx, seq_start_idx, seq_end_idx,
                    v, do, dvd, kt, A, beta, h, dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo,
                )

    else:

        @T.prim_func
        def tilelang_bwd_post_a_kernel(
            v: T.Tensor(v_shape, dtype=qkva_dtype),
            do: T.Tensor(v_shape, dtype=qkva_dtype),
            dvd: T.Tensor(v_shape, dtype=qkva_dtype),
            kt: T.Tensor(k_shape, dtype=qkva_dtype),
            A: T.Tensor(a_shape, dtype=qkva_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            h: T.Tensor(h_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            dbeta_w: T.Tensor(b_shape, dtype=accum_dtype),
            dAqk: T.Tensor(a_shape, dtype=qkva_dtype),
            dL: T.Tensor(a_shape, dtype=qkva_dtype),
            vd_out: T.Tensor(v_shape, dtype=qkva_dtype),
            dv_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            vd_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                if single_batch:
                    kernel_body(
                        0, 0, bc, bh, bc, 0, num_tokens,
                        v, do, dvd, kt, A, beta, h, dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo,
                    )
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

                    kernel_body(
                        bb, bb, chunk_idx, bh, chunk_idx, seq_start_idx, seq_end_idx,
                        v, do, dvd, kt, A, beta, h, dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo,
                    )

    return tilelang_bwd_post_a_kernel


# ---------------------------------------------------------------------------
# K3b
# ---------------------------------------------------------------------------

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_bwd_post_b(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    seqlen_dtype,
    is_varlen,
    single_batch,
    state_v_first,
    split,
    use_qk,
):
    """dq_p / dk_p are written as bf16 (dg_p stays fp32; K4 reads them with a
    cast; they never enter dg').  ``split``: dX and Vd enter the GEMMs as hi + lo bf16 pairs (dv_lo / vd_lo from K3a). ``use_qk``: the dg' terms use
    fp32 q / k (recomputed q~ = q 2^g, k~ = k 2^g, kbar = k 2^(gL-g)) instead of the bf16 q~ / k~ / kbar tensors (which
    are then unused; q, k are [B, T, Hg, K]). Both off = the original bf16 kernel."""
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    h_batch = T.dynamic("h_batch")
    h_chunks = T.dynamic("h_chunks")
    block_S = chunk_size
    KB = 32  # channels per K block
    NKB = DK // KB

    k_shape = (data_batch_size, num_tokens, H, DK)
    qk_shape = (data_batch_size, num_tokens, Hg, DK)
    v_shape = (data_batch_size, num_tokens, H, DV)
    h_shape = (h_batch, h_chunks, H, DV, DK) if state_v_first else (h_batch, h_chunks, H, DK, DV)
    gl_shape = (h_batch, h_chunks, H, DK)
    p_dtype = qkva_dtype

    @T.macro
    def kernel_body(
        bb,
        hb,
        gid,
        bh,
        bhg,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        do,
        dv,
        vd,
        h,
        dh,
        gl,
        qt,
        kt,
        kb,
        g,
        dv_lo,
        vd_lo,
        q,
        k,
        dq_p,
        dk_p,
        dg_p,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        last_row = T.if_then_else(right <= seq_end_idx, block_S - 1, seq_end_idx - left - 1)

        do_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
        dx_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
        vd_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
        if split:
            dxl_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            vdl_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
        h_shared = T.alloc_shared((DV, KB) if state_v_first else (KB, DV), dtype=qkva_dtype)
        dh_shared = T.alloc_shared((DV, KB) if state_v_first else (KB, DV), dtype=qkva_dtype)
        if use_qk:
            q_s = T.alloc_shared((block_S, KB), dtype=qkva_dtype)
            k_s = T.alloc_shared((block_S, KB), dtype=qkva_dtype)
            gL_s = T.alloc_shared((KB,), dtype=accum_dtype, scope="shared")
            g_f = T.alloc_fragment((block_S, KB), dtype=accum_dtype)
        else:
            qt_s = T.alloc_shared((block_S, KB), dtype=qkva_dtype)
            kt_s = T.alloc_shared((block_S, KB), dtype=qkva_dtype)
            kb_s = T.alloc_shared((block_S, KB), dtype=qkva_dtype)
            g_s = T.alloc_shared((block_S, KB), dtype=accum_dtype)
        gl_s = T.alloc_shared((KB,), dtype=accum_dtype, scope="shared")
        cs_shared = T.alloc_shared((KB,), dtype=accum_dtype, scope="shared")
        dgl_shared = T.alloc_shared((KB,), dtype=accum_dtype, scope="shared")

        dqt_f = T.alloc_fragment((block_S, KB), dtype=accum_dtype)
        dkt_f = T.alloc_fragment((block_S, KB), dtype=accum_dtype)
        dkb_f = T.alloc_fragment((block_S, KB), dtype=accum_dtype)
        prod_f = T.alloc_fragment((block_S, KB), dtype=accum_dtype)
        cs_f = T.alloc_fragment((KB,), dtype=accum_dtype)
        hd_f = T.alloc_fragment((DV, KB) if state_v_first else (KB, DV), dtype=accum_dtype)
        dgl_f = T.alloc_fragment((KB,), dtype=accum_dtype)

        # ---- full-V operand tiles of the chunk (zero padded past the sequence end) ----
        if right <= seq_end_idx:
            T.async_copy(do[bb, left:right, bh, 0:DV], do_shared)
            T.async_copy(dv[bb, left:right, bh, 0:DV], dx_shared)
            T.async_copy(vd[bb, left:right, bh, 0:DV], vd_shared)
            if split:
                T.async_copy(dv_lo[bb, left:right, bh, 0:DV], dxl_shared)
                T.async_copy(vd_lo[bb, left:right, bh, 0:DV], vdl_shared)
            T.ptx_wait_group(0)
        else:
            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < seq_end_idx:
                    do_shared[j_s, j_v] = do[bb, left + j_s, bh, j_v]
                    dx_shared[j_s, j_v] = dv[bb, left + j_s, bh, j_v]
                    vd_shared[j_s, j_v] = vd[bb, left + j_s, bh, j_v]
                else:
                    do_shared[j_s, j_v] = 0
                    dx_shared[j_s, j_v] = 0
                    vd_shared[j_s, j_v] = 0
            if split:
                for j_s, j_v in T.Parallel(block_S, DV):
                    if left + j_s < seq_end_idx:
                        dxl_shared[j_s, j_v] = dv_lo[bb, left + j_s, bh, j_v]
                        vdl_shared[j_s, j_v] = vd_lo[bb, left + j_s, bh, j_v]
                    else:
                        dxl_shared[j_s, j_v] = 0
                        vdl_shared[j_s, j_v] = 0

        for i_k in T.serial(NKB):
            k0 = i_k * KB
            # ---- K-block inputs ----
            if right <= seq_end_idx:
                if use_qk:
                    T.async_copy(q[bb, left:right, bhg, k0 : k0 + KB], q_s)
                    T.async_copy(k[bb, left:right, bhg, k0 : k0 + KB], k_s)
                else:
                    T.async_copy(qt[bb, left:right, bh, k0 : k0 + KB], qt_s)
                    T.async_copy(kt[bb, left:right, bh, k0 : k0 + KB], kt_s)
                    T.async_copy(kb[bb, left:right, bh, k0 : k0 + KB], kb_s)
                    T.async_copy(g[bb, left:right, bh, k0 : k0 + KB], g_s)
                if state_v_first:
                    T.async_copy(h[hb, gid, bh, 0:DV, k0 : k0 + KB], h_shared)
                    T.async_copy(dh[hb, gid, bh, 0:DV, k0 : k0 + KB], dh_shared)
                else:
                    T.async_copy(h[hb, gid, bh, k0 : k0 + KB, 0:DV], h_shared)
                    T.async_copy(dh[hb, gid, bh, k0 : k0 + KB, 0:DV], dh_shared)
                T.ptx_wait_group(0)
                if use_qk:
                    T.copy(g[bb, left:right, bh, k0 : k0 + KB], g_f)
            else:
                if use_qk:
                    for j_s, j_c in T.Parallel(block_S, KB):
                        if left + j_s < seq_end_idx:
                            q_s[j_s, j_c] = q[bb, left + j_s, bhg, k0 + j_c]
                            k_s[j_s, j_c] = k[bb, left + j_s, bhg, k0 + j_c]
                            g_f[j_s, j_c] = g[bb, left + j_s, bh, k0 + j_c]
                        else:
                            q_s[j_s, j_c] = 0
                            k_s[j_s, j_c] = 0
                            g_f[j_s, j_c] = 0
                else:
                    for j_s, j_c in T.Parallel(block_S, KB):
                        if left + j_s < seq_end_idx:
                            qt_s[j_s, j_c] = qt[bb, left + j_s, bh, k0 + j_c]
                            kt_s[j_s, j_c] = kt[bb, left + j_s, bh, k0 + j_c]
                            kb_s[j_s, j_c] = kb[bb, left + j_s, bh, k0 + j_c]
                            g_s[j_s, j_c] = g[bb, left + j_s, bh, k0 + j_c]
                        else:
                            qt_s[j_s, j_c] = 0
                            kt_s[j_s, j_c] = 0
                            kb_s[j_s, j_c] = 0
                            g_s[j_s, j_c] = 0
                if state_v_first:
                    T.copy(h[hb, gid, bh, 0:DV, k0 : k0 + KB], h_shared)
                    T.copy(dh[hb, gid, bh, 0:DV, k0 : k0 + KB], dh_shared)
                else:
                    T.copy(h[hb, gid, bh, k0 : k0 + KB, 0:DV], h_shared)
                    T.copy(dh[hb, gid, bh, k0 : k0 + KB, 0:DV], dh_shared)
            for j_c in T.Parallel(KB):
                gl_s[j_c] = gl[hb, gid, bh, k0 + j_c]
            if use_qk:
                for j_c in T.Parallel(KB):
                    gL_s[j_c] = g[bb, left + last_row, bh, k0 + j_c]
            T.fence_proxy_async()

            # dq~ = dO S^T ; dk~ = dX S^T (sign applied below) ; dkbar = Vd dS'^T   (contraction over V; dX and Vd as
            # hi + lo when split)
            if state_v_first:
                T.gemm(do_shared, h_shared, dqt_f, clear_accum=True, policy=_POLICY)
                T.gemm(dx_shared, h_shared, dkt_f, clear_accum=True, policy=_POLICY)
                T.gemm(vd_shared, dh_shared, dkb_f, clear_accum=True, policy=_POLICY)
                if split:
                    T.gemm(dxl_shared, h_shared, dkt_f, clear_accum=False, policy=_POLICY)
                    T.gemm(vdl_shared, dh_shared, dkb_f, clear_accum=False, policy=_POLICY)
            else:
                T.gemm(do_shared, h_shared, dqt_f, transpose_B=True, clear_accum=True, policy=_POLICY)
                T.gemm(dx_shared, h_shared, dkt_f, transpose_B=True, clear_accum=True, policy=_POLICY)
                T.gemm(vd_shared, dh_shared, dkb_f, transpose_B=True, clear_accum=True, policy=_POLICY)
                if split:
                    T.gemm(dxl_shared, h_shared, dkt_f, transpose_B=True, clear_accum=False, policy=_POLICY)
                    T.gemm(vdl_shared, dh_shared, dkb_f, transpose_B=True, clear_accum=False, policy=_POLICY)

            # column sums over rows of kbar (.) dkbar and dgl_k = sum_v S_kv dS'_kv
            if use_qk:
                for j_s, j_c in T.Parallel(block_S, KB):
                    prod_f[j_s, j_c] = (
                        k_s[j_s, j_c] * T.exp2(_cap(gL_s[j_c] - g_f[j_s, j_c], 0.0)) * dkb_f[j_s, j_c]
                    )
            else:
                for j_s, j_c in T.Parallel(block_S, KB):
                    prod_f[j_s, j_c] = kb_s[j_s, j_c] * dkb_f[j_s, j_c]
            T.reduce_sum(prod_f, cs_f, dim=0, clear=True)
            for j_c in T.Parallel(KB):
                cs_shared[j_c] = cs_f[j_c]
            if state_v_first:
                for j_v, j_c in T.Parallel(DV, KB):
                    hd_f[j_v, j_c] = h_shared[j_v, j_c]
                for j_v, j_c in T.Parallel(DV, KB):
                    hd_f[j_v, j_c] = hd_f[j_v, j_c] * dh_shared[j_v, j_c]
                T.reduce_sum(hd_f, dgl_f, dim=0, clear=True)
            else:
                for j_c, j_v in T.Parallel(KB, DV):
                    hd_f[j_c, j_v] = h_shared[j_c, j_v]
                for j_c, j_v in T.Parallel(KB, DV):
                    hd_f[j_c, j_v] = hd_f[j_c, j_v] * dh_shared[j_c, j_v]
                T.reduce_sum(hd_f, dgl_f, dim=1, clear=True)
            for j_c in T.Parallel(KB):
                dgl_shared[j_c] = dgl_f[j_c]

            # ---- epilogue (exp2 arguments <= 0; the caps are identities for valid rows) ----
            if use_qk:
                for j_s, j_c in T.Parallel(block_S, KB):
                    gm = T.exp2(_cap(g_f[j_s, j_c], 0.0))
                    el = T.exp2(_cap(gL_s[j_c] - g_f[j_s, j_c], 0.0))
                    dqt = scale * dqt_f[j_s, j_c]
                    dktv = dkt_f[j_s, j_c] * -1.0
                    if left + j_s < seq_end_idx:
                        dq_p[bb, left + j_s, bh, k0 + j_c] = dqt * gm
                        dk_p[bb, left + j_s, bh, k0 + j_c] = dktv * gm + dkb_f[j_s, j_c] * el
                        dg_p[bb, left + j_s, bh, k0 + j_c] = (
                            gm * (q_s[j_s, j_c] * dqt + k_s[j_s, j_c] * dktv)
                            - el * k_s[j_s, j_c] * dkb_f[j_s, j_c]
                            + T.if_then_else(
                                j_s == last_row, cs_shared[j_c] + gl_s[j_c] * dgl_shared[j_c], 0.0
                            )
                        )
            else:
                for j_s, j_c in T.Parallel(block_S, KB):
                    gm = T.exp2(_cap(g_s[j_s, j_c], 0.0))
                    el = T.exp2(_cap(g_s[last_row, j_c] - g_s[j_s, j_c], 0.0))
                    dqt = scale * dqt_f[j_s, j_c]
                    dktv = dkt_f[j_s, j_c] * -1.0
                    if left + j_s < seq_end_idx:
                        dq_p[bb, left + j_s, bh, k0 + j_c] = dqt * gm
                        dk_p[bb, left + j_s, bh, k0 + j_c] = dktv * gm + dkb_f[j_s, j_c] * el
                        dg_p[bb, left + j_s, bh, k0 + j_c] = (
                            qt_s[j_s, j_c] * dqt
                            + kt_s[j_s, j_c] * dktv
                            - kb_s[j_s, j_c] * dkb_f[j_s, j_c]
                            + T.if_then_else(
                                j_s == last_row, cs_shared[j_c] + gl_s[j_c] * dgl_shared[j_c], 0.0
                            )
                        )

    if is_varlen:

        @T.prim_func
        def tilelang_bwd_post_b_kernel(
            do: T.Tensor(v_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            vd: T.Tensor(v_shape, dtype=qkva_dtype),
            h: T.Tensor(h_shape, dtype=qkva_dtype),
            dh: T.Tensor(h_shape, dtype=qkva_dtype),
            gl: T.Tensor(gl_shape, dtype=accum_dtype),
            qt: T.Tensor(k_shape, dtype=qkva_dtype),
            kt: T.Tensor(k_shape, dtype=qkva_dtype),
            kb: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(k_shape, dtype=accum_dtype),
            dv_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            vd_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            q: T.Tensor(qk_shape, dtype=qkva_dtype),
            k: T.Tensor(qk_shape, dtype=qkva_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            dq_p: T.Tensor(k_shape, dtype=p_dtype),
            dk_p: T.Tensor(k_shape, dtype=p_dtype),
            dg_p: T.Tensor(k_shape, dtype=accum_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    0, 0, bc, bh, bhg, chunk_idx, seq_start_idx, seq_end_idx,
                    do, dv, vd, h, dh, gl, qt, kt, kb, g, dv_lo, vd_lo, q, k, dq_p, dk_p, dg_p,
                )

    else:

        @T.prim_func
        def tilelang_bwd_post_b_kernel(
            do: T.Tensor(v_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            vd: T.Tensor(v_shape, dtype=qkva_dtype),
            h: T.Tensor(h_shape, dtype=qkva_dtype),
            dh: T.Tensor(h_shape, dtype=qkva_dtype),
            gl: T.Tensor(gl_shape, dtype=accum_dtype),
            qt: T.Tensor(k_shape, dtype=qkva_dtype),
            kt: T.Tensor(k_shape, dtype=qkva_dtype),
            kb: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(k_shape, dtype=accum_dtype),
            dv_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            vd_lo: T.Tensor(v_shape, dtype=qkva_dtype),
            q: T.Tensor(qk_shape, dtype=qkva_dtype),
            k: T.Tensor(qk_shape, dtype=qkva_dtype),
            dq_p: T.Tensor(k_shape, dtype=p_dtype),
            dk_p: T.Tensor(k_shape, dtype=p_dtype),
            dg_p: T.Tensor(k_shape, dtype=accum_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                if single_batch:
                    kernel_body(
                        0, 0, bc, bh, bhg, bc, 0, num_tokens,
                        do, dv, vd, h, dh, gl, qt, kt, kb, g, dv_lo, vd_lo, q, k, dq_p, dk_p, dg_p,
                    )
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
                        bb, bb, chunk_idx, bh, bhg, chunk_idx, seq_start_idx, seq_end_idx,
                        do, dv, vd, h, dh, gl, qt, kt, kb, g, dv_lo, vd_lo, q, k, dq_p, dk_p, dg_p,
                    )

    return tilelang_bwd_post_b_kernel


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _grid_info(cu_seqlens, B, Tn, chunk_size):
    if cu_seqlens is None:
        return B * tilelang.cdiv(Tn, chunk_size), "int32", False, None
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return len(chunk_indices), cu_seqlens.dtype, True, chunk_indices


def bwd_post_a_kda(do, v, h, dvd, A, beta, kt, cu_seqlens: Optional[torch.LongTensor] = None,
                   state_v_first: bool = False, chunk_size: int = 64, return_lo: bool = False):
    """K3a. Returns (dv, dbeta_w, dAqk, dL, Vd); with ``return_lo`` also (dv_lo, vd_lo), the bf16 residuals
    dX - dv and Vd_fp32 - Vd (K3b's hi/lo operands). See the module docstring."""
    B, Tn, H, V = v.shape
    K = kt.shape[-1]
    assert K == V == 128 and chunk_size == 64
    assert do.dtype == v.dtype == h.dtype == dvd.dtype == A.dtype == kt.dtype == torch.bfloat16
    assert beta.dtype == torch.float32
    for t in (do, v, h, dvd, A, beta, kt):
        assert t.is_contiguous()
    num_chunks, seqlen_dtype, is_varlen, chunk_indices = _grid_info(cu_seqlens, B, Tn, chunk_size)
    dv = torch.empty_like(v)
    vd_out = torch.empty_like(v)
    dv_lo, vd_lo = torch.empty_like(v), torch.empty_like(v)
    dbeta_w = torch.empty_like(beta)
    dAqk, dL = torch.empty_like(A), torch.empty_like(A)
    kernel = tilelang_bwd_post_a(
        H, K, V, chunk_size, accum_dtype="float32", qkva_dtype=v.dtype, seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen, state_v_first=state_v_first, single_batch=(not is_varlen and B == 1),
    )
    if is_varlen:
        kernel(v, do, dvd, kt, A, beta, h, cu_seqlens, chunk_indices, dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo)
    else:
        kernel(v, do, dvd, kt, A, beta, h, dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo, num_chunks)
    if return_lo:
        return dv, dbeta_w, dAqk, dL, vd_out, dv_lo, vd_lo
    return dv, dbeta_w, dAqk, dL, vd_out


def bwd_post_b_kda(do, dv, vd, h, dh, gl, qt, kt, kb, g_c, scale: Optional[float] = None,
                   cu_seqlens: Optional[torch.LongTensor] = None, state_v_first: bool = False,
                   chunk_size: int = 64, dv_lo=None, vd_lo=None, q=None, k=None):
    """K3b. Returns (dq_p, dk_p, dg_p) fp32 [B, T, HV, K]. ``dv_lo`` / ``vd_lo`` (K3a ``return_lo``): use dX and Vd as
    hi + lo bf16 pairs. ``q`` / ``k`` [B, T, HK, K] bf16: recompute q~ / k~ / kbar in fp32 for the dg' terms (qt / kt /
    kb are then not read). Without them the original all-bf16 kernel runs. dq_p / dk_p are returned as bf16."""
    B, Tn, H, V = do.shape
    K = qt.shape[-1]
    assert K == V == 128 and chunk_size == 64
    assert do.dtype == dv.dtype == vd.dtype == h.dtype == dh.dtype == qt.dtype == kt.dtype == kb.dtype == torch.bfloat16
    assert g_c.dtype == torch.float32 and gl.dtype == torch.float32
    assert (dv_lo is None) == (vd_lo is None) and (q is None) == (k is None)
    split, use_qk = dv_lo is not None, q is not None
    for t in (do, dv, vd, h, dh, gl, qt, kt, kb, g_c) + ((dv_lo, vd_lo) if split else ()) + ((q, k) if use_qk else ()):
        assert t.is_contiguous()
    if not split:
        dv_lo, vd_lo = dv, vd  # unused placeholders
    if not use_qk:
        q, k = qt, kt
    scale = K ** -0.5 if scale is None else scale
    num_chunks, seqlen_dtype, is_varlen, chunk_indices = _grid_info(cu_seqlens, B, Tn, chunk_size)
    dg_p = torch.empty_like(g_c)
    dq_p, dk_p = (torch.empty_like(g_c, dtype=torch.bfloat16) for _ in range(2))
    kernel = tilelang_bwd_post_b(
        H, q.shape[2], K, V, chunk_size, float(scale), accum_dtype="float32", qkva_dtype=do.dtype,
        seqlen_dtype=seqlen_dtype, is_varlen=is_varlen, state_v_first=state_v_first,
        single_batch=(not is_varlen and B == 1), split=split, use_qk=use_qk,
    )
    if is_varlen:
        kernel(do, dv, vd, h, dh, gl, qt, kt, kb, g_c, dv_lo, vd_lo, q, k, cu_seqlens, chunk_indices, dq_p, dk_p, dg_p)
    else:
        kernel(do, dv, vd, h, dh, gl, qt, kt, kb, g_c, dv_lo, vd_lo, q, k, dq_p, dk_p, dg_p, num_chunks)
    return dq_p, dk_p, dg_p


def bwd_post_kda(do, v, h, dh, dvd, A, beta, qt, kt, kb, gl, g_c, scale: Optional[float] = None,
                 cu_seqlens: Optional[torch.LongTensor] = None, state_v_first: bool = False, chunk_size: int = 64,
                 q=None, k=None):
    """K3a + K3b (same I/O as ``torch_ref.bwd_post_torch`` except: dgl is computed inside, dh0 / dgl inputs are not
    needed). dX / Vd are passed to K3b as hi + lo pairs; with ``q`` / ``k`` the dg' terms use fp32 q~ / k~ / kbar.
    Returns dict(dv, dbeta_w, dAqk, dL, Vd, dq_p, dk_p, dg_p)."""
    dv, dbeta_w, dAqk, dL, vd, dv_lo, vd_lo = bwd_post_a_kda(
        do, v, h, dvd, A, beta, kt, cu_seqlens, state_v_first, chunk_size, return_lo=True)
    dq_p, dk_p, dg_p = bwd_post_b_kda(do, dv, vd, h, dh, gl, qt, kt, kb, g_c, scale, cu_seqlens, state_v_first,
                                      chunk_size, dv_lo=dv_lo, vd_lo=vd_lo, q=q, k=k)
    return dict(dv=dv, dbeta_w=dbeta_w, dAqk=dAqk, dL=dL, Vd=vd, dq_p=dq_p, dk_p=dk_p, dg_p=dg_p)
