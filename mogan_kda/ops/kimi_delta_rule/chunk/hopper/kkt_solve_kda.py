# Derived from mogan_kda/ops/gated_delta_rule/chunk/hopper/kkt_solve.py
# (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT License).
"""Per-channel-gate (KDA) ``kkt_solve``: intra-chunk Akk^-1 and Aqk of a 64-token chunk.

For every chunk and v-head (q/k head = hv // (HV // HK); i, j chunk-local rows; g = chunk-local cumsum of
the per-token log-decay)::

    Gamma_ij = sum_c k_ic k_jc exp(g_ic - g_jc)                     (i > j)
    Akk      = (I + StrictLower(diag(beta) Gamma))^-1               -> A   [B, T, HV, 64]
    Aqk_ij   = scale * sum_c q_ic k_jc exp(g_ic - g_jc)   (i >= j)  -> Aqk [B, T, HV, 64]  (0 above the diagonal)

Both outputs are bf16 (fp32 accumulation). Unlike the scalar-gate kernel, the gate sits inside the inner
product, so ``Gamma`` cannot be factored as ``D (K K^T) D^-1``; exp(g_i - g_j) is anchored instead.

Gate convention: ``g`` is the chunk-local cumsum in the **log2 domain**, i.e.
``chunk_local_cumsum(g_natural_log, scale=1/ln2)``; the kernel uses exp2 directly. Contract: the
per-token log-decay is >= lower_bound = -5 (natural log), i.e. >= -5/ln2 = -7.21 in log2 units.

Sub-block anchoring (4 sub-blocks of 16 rows, every exponent argument below is <= 0 except the marked one)
  * off-diagonal sub-blocks (row block I > column block J): anchor a = first row of block I.
    q~_i = q_i exp2(g_i - g_a), k~_i = k_i exp2(g_i - g_a) (rows i in I), k^_j = k_j exp2(g_a - g_j) (rows j
    in J < I). All factors are <= 1 -> bf16 tensor-core GEMMs (fp32 accumulate).
  * diagonal 16x16 sub-blocks: FLA anchors these at the middle row with bf16 tensor cores (batch-invariant,
    but not bitwise prefix-invariant, see fla#1240). We require bitwise prefix invariance, so we anchor at
    row 0 of the sub-block. Worst case exp2(g_0 - g_j) = 2^(15 * 7.21) = e^75 < e^88 (fp32 max) under the
    contract above. fp32 operands on tf32 tensor cores (one 64x64 GEMM per K-quarter, entries of other
    sub-blocks discarded); operands are rounded to tf32 (10-bit mantissa), the
    outputs are bf16 anyway. Aqk[i, i] = scale * q_i . k_i is gate independent (up to tf32 rounding).

Prefix invariance (bitwise): no value in row i depends on any token > i.
  * every anchor row is <= the rows that use it (off-diagonal: first row of the row block I; diagonal: row 0 of
    the sub-block) and the chunk-local cumsum is forward, so g rows > i never enter row i;
  * tensor-core GEMM output element (i, j) depends only on row i of the A operand and row j of the B operand;
    the operands of rows >= i never enter it (rows outside the block / columns >= the anchor are zero);
  * partial chunks / rows past the sequence end are zero padded (loads) and masked (stores);
  * the inversion stage is forward substitution (row i uses rows <= i) plus row-independent GEMM merges.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from mogan_kda.utils import prepare_chunk_indices

RCP_LN2 = 1.4426950408889634

# Layout fallback switch. False: the phase-1 (tf32) and phase-2 (bf16) GEMMs accumulate into the same two 64x64
# fragments; every T.gemm passes policy=FullRow so both lower to the same M-stacked (4 warps along M) layout.
# True: phase 2 runs first into a64/aqk, phase 1 accumulates into its own fragments (own layout), is masked, goes
# through an fp32 smem tile and is added to a64/aqk with a T.Parallel (costs 2 extra fragments of registers).
PHASE1_SEPARATE = False
_POLICY = GemmWarpPolicy.FullRow


@tilelang.jit(
    # out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve_kda(
    H,
    Hg,
    DK,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
    emit_scaled=False,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    SUB = 16  # sub-block rows
    NSUB = block_S // SUB
    KQ = 32  # channels per K-quarter
    NQ = DK // KQ
    NSLOT = 4  # K-slots of the packed off-diagonal operands (3 used: row blocks 1..3)

    q_shape = (data_batch_size, num_tokens, Hg, DK)
    g_shape = (data_batch_size, num_tokens, H, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    # emit_scaled outputs (own dynamic dims: when emit_scaled is False 1-element dummies are passed)
    s_shape = (T.dynamic("scaled_batch"), T.dynamic("scaled_tokens"), H, DK)
    gl_shape = (T.dynamic("gl_batch"), T.dynamic("gl_chunks"), H, DK)

    @T.macro
    def phase1(q_shared, k_shared, g_shared, kd_shared, qd_shared, kdh_shared, acc_a, acc_q, left, seq_end_idx):
        # ---- Phase 1: diagonal 16x16 sub-blocks, ONE 64x64 GEMM per K-quarter on tensor cores (tf32, fp32 accumulate).
        # Every row uses the anchor of its own sub-block, row 0 of that sub-block (never a later row):
        #   kd_i = k_i 2^(g_i - g_a(i)), qd_i = q_i 2^(g_i - g_a(i)), kdh_j = k_j 2^(g_a(j) - g_j)
        # so entry (i, j) with a(i) == a(j) is sum_c k_ic k_jc 2^(g_ic - g_jc) (i > j; exponents of kd/qd <= 0, of kdh
        # <= 15 * 7.21 -> 2^108 < 2^128 under the lower_bound = -5 contract). Entries with a(i) != a(j) are finite garbage
        # and are zeroed right after (before the off-diagonal GEMMs accumulate into the same fragments). FLA anchors
        # these blocks at the middle row with bf16 tensor cores (batch-invariant, not bitwise prefix-invariant,
        # see fla#1240); ours stays prefix invariant: an entry of the GEMM output depends only on row i of A and row j of B.
        T.clear(acc_a)
        T.clear(acc_q)
        for i_q in T.serial(NQ):
            c0 = i_q * KQ
            for j_s, j_c in T.Parallel(block_S, KQ):
                if left + j_s < seq_end_idx:
                    kd_shared[j_s, j_c] = k_shared[j_s, c0 + j_c] * T.exp2(
                        g_shared[j_s, c0 + j_c] - g_shared[(j_s // SUB) * SUB, c0 + j_c]
                    )
                    qd_shared[j_s, j_c] = q_shared[j_s, c0 + j_c] * T.exp2(
                        g_shared[j_s, c0 + j_c] - g_shared[(j_s // SUB) * SUB, c0 + j_c]
                    )
                    kdh_shared[j_s, j_c] = k_shared[j_s, c0 + j_c] * T.exp2(
                        g_shared[(j_s // SUB) * SUB, c0 + j_c] - g_shared[j_s, c0 + j_c]
                    )
                else:
                    kd_shared[j_s, j_c] = 0
                    qd_shared[j_s, j_c] = 0
                    kdh_shared[j_s, j_c] = 0
            T.fence_proxy_async()
            T.gemm(kd_shared, kdh_shared, acc_a, transpose_B=True, policy=_POLICY)
            T.gemm(qd_shared, kdh_shared, acc_q, transpose_B=True, policy=_POLICY)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s // SUB != j_t // SUB:
                acc_a[j_s, j_t] = 0
                acc_q[j_s, j_t] = 0


    @T.macro
    def phase2(q_shared, k_shared, g_shared, ap_shared, bp_shared, acc_a, acc_q, left, seq_end_idx):
        # Phase 2 tiles are structurally zero outside their live region: zero them once. Done AFTER phase 1 on purpose:
        # their lifetime then starts after the phase 1 tiles die and the smem planner can alias the two sets
        # (peak smem ~ raw q/k/g 64 KB + 32 KB of tiles).
        for j_s, j_p in T.Parallel(block_S, NSLOT * KQ):
            ap_shared[j_s, j_p] = 0
            bp_shared[j_s, j_p] = 0

        # ---- Phase 2: off-diagonal sub-blocks (row block I > column block J), anchor = first row of row block I.
        # Row-block-diagonal packing: A' row i (block I in 1..3) holds q~_i / k~_i = x_i 2^(g_i - g_a(I)) in K-slot I - 1
        # and zeros elsewhere; B' row j holds k^_j = k_j 2^(g_a(I) - g_j) in slot I - 1 for every j < 16 I (zero
        # otherwise). One GEMM A' B'^T then yields, for row i in block I, exactly sum_c q~_ic k^_jc (j < 16 I) with
        # no cross-block garbage and no per-block operand rebuild. All factors <= 1 (g non-increasing) -> bf16.
        for i_q in T.serial(NQ):
            c0 = i_q * KQ
            # Extents must be powers of 2 (T.Parallel(48, 3, 32) fails layout inference: "no available layout found"),
            # hence (block_S, NSLOT, KQ) with slot 3 and rows >= 48 masked off by the condition.
            for j_s, s_slot, j_c in T.Parallel(block_S, NSLOT, KQ):
                if (s_slot < NSLOT - 1) and (j_s < SUB * (s_slot + 1)) and (left + SUB * (s_slot + 1) < seq_end_idx) and (left + j_s < seq_end_idx):
                    bp_shared[j_s, s_slot * KQ + j_c] = k_shared[j_s, c0 + j_c] * T.exp2(
                        g_shared[SUB * (s_slot + 1), c0 + j_c] - g_shared[j_s, c0 + j_c]
                    )
            for j_s, j_c in T.Parallel(block_S - SUB, KQ):
                if left + SUB + j_s < seq_end_idx:
                    ap_shared[SUB + j_s, (j_s // SUB) * KQ + j_c] = k_shared[SUB + j_s, c0 + j_c] * T.exp2(
                        g_shared[SUB + j_s, c0 + j_c] - g_shared[SUB * (1 + j_s // SUB), c0 + j_c]
                    )
            T.fence_proxy_async()
            T.gemm(ap_shared, bp_shared, acc_a, transpose_B=True, policy=_POLICY)
            for j_s, j_c in T.Parallel(block_S - SUB, KQ):
                if left + SUB + j_s < seq_end_idx:
                    ap_shared[SUB + j_s, (j_s // SUB) * KQ + j_c] = q_shared[SUB + j_s, c0 + j_c] * T.exp2(
                        g_shared[SUB + j_s, c0 + j_c] - g_shared[SUB * (1 + j_s // SUB), c0 + j_c]
                    )
            T.fence_proxy_async()
            T.gemm(ap_shared, bp_shared, acc_q, transpose_B=True, policy=_POLICY)


    @T.macro
    def kernel_body(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        q,
        k,
        g,
        b,
        a,
        aqk,
        qt,
        kt,
        kb,
        gl,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        g_shared = T.alloc_shared((block_S, DK), dtype=g_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        aqk_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        # Phase 1 (diagonal sub-blocks, tf32 GEMM): fp32 operand tiles of one K-quarter (32 channels).
        # The planner aliases them with the phase 2 tiles (disjoint lifetimes).
        qd_shared = T.alloc_shared((block_S, KQ), dtype=accum_dtype)
        kd_shared = T.alloc_shared((block_S, KQ), dtype=accum_dtype)
        kdh_shared = T.alloc_shared((block_S, KQ), dtype=accum_dtype)
        # Phase 2 (off-diagonal sub-blocks, bf16 GEMM): K is packed as NSLOT = 4 slots of KQ channels; slot s serves
        # row block s + 1 (row-block-diagonal packing, see the module docstring). Slot 3 stays zero.
        ap_shared = T.alloc_shared((block_S, NSLOT * KQ), dtype=qkva_dtype)  # A': k~ then q~ rows
        bp_shared = T.alloc_shared((block_S, NSLOT * KQ), dtype=qkva_dtype)  # B': k^ rows

        aq_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        # PHASE1_SEPARATE fallback only (1x1 dummies otherwise)
        d_shape = (block_S, block_S) if PHASE1_SEPARATE else (1, 1)
        d64_fragment = T.alloc_fragment(d_shape, dtype=accum_dtype)
        dq_fragment = T.alloc_fragment(d_shape, dtype=accum_dtype)
        d64_shared = T.alloc_shared(d_shape, dtype=accum_dtype)
        dq_shared = T.alloc_shared(d_shape, dtype=accum_dtype)

        # inversion stage: unchanged from kkt_solve
        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        # Load q, k, g (cp.async for full chunks; zero padded past the sequence end) and beta
        if right <= seq_end_idx:
            T.async_copy(q[bb, left:right, bhg, 0:DK], q_shared)
            T.async_copy(k[bb, left:right, bhg, 0:DK], k_shared)
            T.async_copy(g[bb, left:right, bh, 0:DK], g_shared)
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[bb, left + j_s, bh]
        else:
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    q_shared[j_s, j_k] = q[bb, left + j_s, bhg, j_k]
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                    g_shared[j_s, j_k] = g[bb, left + j_s, bh, j_k]
                else:
                    q_shared[j_s, j_k] = 0
                    k_shared[j_s, j_k] = 0
                    g_shared[j_s, j_k] = 0
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        if right <= seq_end_idx:
            T.ptx_wait_group(0)

        # NOTE (invalid rows): g of rows past the sequence end is zero padded, so exp2(g_i - g_anchor) could overflow
        # there; every operand element of an invalid row (and every kh element whose anchor row is invalid) is
        # therefore written as zero instead of being computed.

        # ---- emit_scaled: gate-scaled operands of the fused forward, straight from the raw tiles to global
        # (no staging tile). q~ = Q exp2(g), k~ = K exp2(g), kbar = K exp2(g_L - g), gl = exp2(g_L), g_L = g of the
        # LAST VALID row of the chunk. All exponent arguments <= 0; rows past the end are not written.
        if emit_scaled:
            last_row = T.if_then_else(right <= seq_end_idx, block_S - 1, seq_end_idx - left - 1)
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    qt[bb, left + j_s, bh, j_k] = q_shared[j_s, j_k] * T.exp2(g_shared[j_s, j_k])
                    kt[bb, left + j_s, bh, j_k] = k_shared[j_s, j_k] * T.exp2(g_shared[j_s, j_k])
                    kb[bb, left + j_s, bh, j_k] = k_shared[j_s, j_k] * T.exp2(
                        g_shared[last_row, j_k] - g_shared[j_s, j_k]
                    )
            # chunk id, laid out like FlashQLA's h: dense [B, cdiv(T, 64)] -> (bb, chunk_idx);
            # varlen [1, total_chunks] -> (0, bc) with bc = chunk_offsets[seq] + chunk_idx (prepare_chunk_indices order)
            for j_k in T.Parallel(DK):
                gl[bb, bc if is_varlen else chunk_idx, bh, j_k] = T.exp2(g_shared[last_row, j_k])

        if PHASE1_SEPARATE:
            T.clear(a64_fragment)
            T.clear(aqk_fragment)
            phase2(q_shared, k_shared, g_shared, ap_shared, bp_shared, a64_fragment, aqk_fragment, left, seq_end_idx)
            phase1(q_shared, k_shared, g_shared, kd_shared, qd_shared, kdh_shared, d64_fragment, dq_fragment, left, seq_end_idx)
            T.copy(d64_fragment, d64_shared)
            T.copy(dq_fragment, dq_shared)
            for j_s, j_t in T.Parallel(block_S, block_S):
                a64_fragment[j_s, j_t] += d64_shared[j_s, j_t]
                aqk_fragment[j_s, j_t] += dq_shared[j_s, j_t]
        else:
            phase1(q_shared, k_shared, g_shared, kd_shared, qd_shared, kdh_shared, a64_fragment, aqk_fragment, left, seq_end_idx)
            phase2(q_shared, k_shared, g_shared, ap_shared, bp_shared, a64_fragment, aqk_fragment, left, seq_end_idx)

        # Aqk = scale * Lower(Aqk) (diagonal included), bf16
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= j_t:
                aq_shared[j_s, j_t] = aqk_fragment[j_s, j_t] * scale
            else:
                aq_shared[j_s, j_t] = 0
        if right <= seq_end_idx:
            T.copy(aq_shared, aqk[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    aqk[bb, left + j_s, bh, j_t] = aq_shared[j_s, j_t]

        # A = b * Gamma
        for j_s, j_t in T.Parallel(block_S, block_S):
            a64_fragment[j_s, j_t] *= b_shared[j_s]

        # A = I + StrictLower(A)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a64_fragment[j_s, j_t] = 1

        # ---- inversion stage: identical to kkt_solve ----
        # Prepare inversion input
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16) + 1:
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16):
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

        # Diagonal 4x16x16
        T.clear(a16i_row)
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(4, 16):
                    a16i_sum[j_s, k_t] -= (
                        a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                    )
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

        # First level 2x16x16
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                )
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

        # Second level 1x32x32
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s < 16 and k_t >= 16:
                a32i_fragment[j_s, k_s, k_t] = 0
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s >= 16 and k_t < 16:
                a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s // 16 == k_t // 16:
                a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                    j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                ]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if j_s == 0:
                a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
            else:
                a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
        T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
        T.copy(a32o_fragment, a32o_shared)
        T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

        # Combine inversion output
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[
                j_s, k_s, k_t
            ]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[k_s, 32 + k_t] = 0

        # Save A (unmasked)
        if right <= seq_end_idx:
            T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]



    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kda_kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            aqk: T.Tensor(a_shape, dtype=qkva_dtype),
            qt: T.Tensor(s_shape, dtype=qkva_dtype),
            kt: T.Tensor(s_shape, dtype=qkva_dtype),
            kb: T.Tensor(s_shape, dtype=qkva_dtype),
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

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    q,
                    k,
                    g,
                    b,
                    a,
                    aqk,
                    qt,
                    kt,
                    kb,
                    gl,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kda_kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            aqk: T.Tensor(a_shape, dtype=qkva_dtype),
            qt: T.Tensor(s_shape, dtype=qkva_dtype),
            kt: T.Tensor(s_shape, dtype=qkva_dtype),
            kb: T.Tensor(s_shape, dtype=qkva_dtype),
            gl: T.Tensor(gl_shape, dtype=accum_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                bb = T.alloc_var("int32")
                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                # int32 locals evaluated once (a plain expression is re-inlined into every global address)
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    q,
                    k,
                    g,
                    b,
                    a,
                    aqk,
                    qt,
                    kt,
                    kb,
                    gl,
                )

    return tilelang_kkt_solve_kda_kernel


def _check_gate_lower_bound(g, chunk_size, cu_seqlens, lower_bound):
    """Assert the per-token log2-decay implied by the chunk-local cumsum ``g`` is >= lower_bound (natural log)."""
    T_ = g.shape[1]
    decay = g.clone()
    decay[:, 1:] -= g[:, :-1]  # per-token decay = cumsum difference ...
    if cu_seqlens is None:
        starts = torch.arange(0, T_, chunk_size, device=g.device)[None].expand(g.shape[0], -1)
        decay[:, starts[0]] = g[:, starts[0]]  # ... except at chunk starts, where it is the cumsum itself
    else:
        idx = prepare_chunk_indices(cu_seqlens, chunk_size).long()
        starts = cu_seqlens.long()[idx[:, 0]] + idx[:, 1] * chunk_size
        decay[:, starts] = g[:, starts]
    worst = decay.min().item()
    bound = lower_bound * RCP_LN2 - 1e-3  # g is log2 domain
    assert worst >= bound, (
        f"per-token log2-decay {worst:.4f} < lower_bound {lower_bound} * 1/ln2 = {lower_bound * RCP_LN2:.4f}: "
        "the anchored exponents can overflow (contract: per-token log-decay >= lower_bound)"
    )


def kkt_solve_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    check_lower_bound: bool = False,
    lower_bound: float = -5.0,
    emit_scaled: bool = False,
):
    """Returns ``(A, Aqk)``, both bf16 ``[B, T, HV, 64]`` (see the module docstring for the math).

    ``emit_scaled=True`` additionally returns ``(A, Aqk, qt, kt, kb, gl)`` for ``fused_fwd_kda``:
    qt = Q exp2(g), kt = K exp2(g), kb = K exp2(g_L - g)  (bf16 ``[B, T, HV, 128]``, zero past the sequence end)
    and gl = exp2(g_L) (fp32 ``[B, NC, HV, 128]``, g_L = g of the last valid row of each chunk); NC layout like
    FlashQLA's ``h``: dense ``ceil(T / 64)`` per batch row, varlen ``[1, total_chunks]``.

    q, k: [B, T, HK, 128] bf16;  g: [B, T, HV, 128] fp32 chunk-local cumsum in the LOG2 domain
    (``chunk_local_cumsum(g, scale=1/ln2)``);  beta: [B, T, HV] fp32.  HV % HK == 0 (GVA).
    ``check_lower_bound`` (default off, costs a few torch ops + a sync) asserts that the per-token decay
    implied by ``g`` is >= ``lower_bound`` (natural log, as in the contract of the anchoring); the caller
    that produces raw g is otherwise responsible for the contract.
    """
    batch_size, num_tokens, Hg, K = q.shape
    _, _, H = beta.shape
    assert K == 128
    assert chunk_size == 64
    assert k.shape == q.shape and g.shape == (batch_size, num_tokens, H, K) and H % Hg == 0
    assert g.dtype == torch.float32 and g.is_contiguous()
    scale = K ** -0.5 if scale is None else scale
    if check_lower_bound:
        _check_gate_lower_bound(g, chunk_size, cu_seqlens, lower_bound)

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty((batch_size, num_tokens, H, chunk_size), dtype=q.dtype, device=q.device)
    aqk = torch.empty_like(a)
    if emit_scaled:
        qt, kt, kb = (torch.empty((batch_size, num_tokens, H, K), dtype=q.dtype, device=q.device) for _ in range(3))
        if is_varlen:
            gl = torch.empty((1, len(chunk_indices), H, K), dtype=torch.float32, device=q.device)
        else:
            gl = torch.empty((batch_size, tilelang.cdiv(num_tokens, chunk_size), H, K), dtype=torch.float32, device=q.device)
    else:  # 1-element dummies: the kernel never touches them
        qt = kt = kb = torch.empty((1, 1, H, K), dtype=q.dtype, device=q.device)
        gl = torch.empty((1, 1, H, K), dtype=torch.float32, device=q.device)

    kernel = tilelang_kkt_solve_kda(
        H,
        Hg,
        K,
        chunk_size,
        float(scale),
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
        emit_scaled=emit_scaled,
    )
    if is_varlen:
        kernel(q, k, g, beta, cu_seqlens, chunk_indices, a, aqk, qt, kt, kb, gl)
    else:
        kernel(q, k, g, beta, a, aqk, qt, kt, kb, gl, num_chunks)

    if emit_scaled:
        return a, aqk, qt, kt, kb, gl
    return a, aqk
