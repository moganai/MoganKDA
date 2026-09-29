# Derived from mogan_kda/ops/gated_delta_rule/chunk/hopper/kkt_solve.py and kkt_solve_kda.py
# (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT License).
"""K4 of the KDA backward: intra-chunk gradients through Gamma and Aqk (per 64-token chunk and v-head).

Same math and I/O as ``torch_ref.bwd_intra_torch`` (i, j chunk-local rows, c channel, E_ij = 2^(g_i - g_j),
g = chunk-local log2 cumsum). With Wq = scale * dAqk and dL, both taken STRICTLY lower (i > j)::

    dq_i  = dq_p_i + sum_{j<i} Wq_ij k_j E_ij + scale dAqk_ii k_i
    T_i   = sum_{j<i} dL_ij k_j E_ij                       dbeta_i = sum_c k_ic T_ic
    dk_i  = dk_p_i + beta_i T_i + sum_{l>i} (Wq_li q_l + beta_l dL_li k_l) E_li + scale dAqk_ii q_i
    dg'_i = dg_p_i + q_i (row term of dq) + k_i (beta_i T_i) - k_i (column term of dk)          (log2 units; the
            ln2 / 1/ln2 factors cancel, K5 turns dg' into the natural-log gradient by a reverse cumsum)

The diagonal element Aqk_ii = scale q_i.k_i is gate independent: it only enters through the separate gate-free adds
``dq_i += scale dAqk_ii k_i``, ``dk_i += scale dAqk_ii q_i`` and never touches dg' (structural zero: every gate-carrying
term reads the strictly lower part only; the diagonal comes from its own vector).

I/O (dtypes): q, k bf16 [B,T,HK,128]; g fp32 [B,T,HV,128] (log2 chunk-local cumsum, contract: per-token decay >= -5
natural log); beta fp32 [B,T,HV]; dAqk, dL bf16 [B,T,HV,64] (chunk-local row i, column j; only the strictly lower part
and the dAqk diagonal are read). Optional fp32 partials dq_p, dk_p, dg_p [B,T,HV,128] (K3b) are added. Outputs are FRESH
fp32 dq, dk (per v-head, K5 sums the GVA group), dg (= dg', before the reverse cumsum) and dbeta_g [B,T,HV].

=============================== structure ===============================
Anchoring per 16-row block (anchor a(B) = first row of block B <= every row of it). E_ij = 2^(g_i - g_a) 2^(g_a - g_j):
  * row side (output row i, summed j, anchor of j's block J): B'[j, (J, c)] = k_jc 2^(g_aJ,c - g_jc), live only in column
    block J = b(j); out[i, (J, c)] = sum_{j in J} M_ij B'[j, (J, c)] for M = dAqk, dL (plain [64,64] tiles, no packing):
    ONE GEMM (M=64, N=4*16, K=64) per 16-channel pass; combined in fp32 with 2^(g_i - g_aJ) (J <= b(i); J > b(i) is an
    exact 0);
  * column side (output row j, summed i, anchor of i's block I): B''[i, (I, c)] = scale q_ic 2^(g_ic - g_aI,c) (rows of
    dAqk^T) resp. beta_i k_ic 2^(g_ic - g_aI,c) (dL^T), factors <= 1; combined with 2^(g_aI,c - g_jc) (I >= b(j)).
Precision: B operands computed in fp32 and split hi = bf16(x), lo = bf16(x - hi) (A operands are exact bf16): 2^-17.
Prefix invariance (bitwise): every anchor used for a NON-zero dA entry is <= the row of the summed index; blocks whose rows
are all >= P have M == 0 exactly (do = 0 there) and contribute exact 0 x finite; fixed accumulation order; no atomics.

=============================== scheduling ===============================
Never use T.if_then_else on per-element values in big loops (use T.min / T.max / mask multiplies) and no manual T.vectorized
(plain T.Parallel auto-vectorises, also into swizzled wgmma tiles and with row-dependent block offsets).
1. OVERLAP. Two hi/lo B-tile pairs and asynchronous WGMMA: T.wgmma_gemm never waits, T.wait_wgmma(n) waits until at most n
   committed wgmma groups (= T.wgmma_gemm calls) are still in flight.
   Per pass: build row operands (pair0) -> issue 4 GEMMs -> build q operands (pair1) while they run -> issue 2
   -> wait(2): the row GEMMs are done, pair0 is free, f_rq / f_rt complete -> build beta*k operands (pair0) -> issue 2
   -> combine the row terms (reads f_rq / f_rt) while the column GEMMs run -> wait(0) -> combine the column term, outputs.
2. LESS ELEMENT WORK. The per-element exp2 factors are computed ONCE per pass: e1 = 2^(g_i - g_a(b(i))) (<= 1) and
   ep = 2^(g_a(b(j)) - g_j) (>= 1, cap 127) into two [64,16] fp32 tiles; all B builds are plain multiplies of them (k*ep,
   scale*q*e1, beta*k*e1). The combines need 2^(g_i - g_aJ) = e1_i * TB[b(i), J] and 2^(g_aI - g_j) = TC[I, b(j)] * ep_j with
   tiny [16,16] tables TB / TC (entries of structurally zero blocks are 0, so no capped garbage): the combine loops contain
   no exp2 at all. Per element per pass: 2 exp2.
3. SCALAR 16-BIT STORES. The strictly-lower masks of the dA tiles are arithmetic (mask multiply by clamp(i - j, 0, 1)), so the
   per-element predicate no longer splits the stores. The column GEMMs read the row-major dA tiles with transpose_A=True
   (no transposed copies).
4. Partials: dq_p / dk_p arrive as bf16 (dg_p fp32) and are staged through smem with cp.async at the top of a pass; dq / dk are
   written as bf16, dg / dbeta in fp32.
Shared memory (KB): dA tiles 2 x 8, B tiles 2 pairs x 2 x 8 = 32, fp32 staging 16, rq / t / c 12, q / k slices 4, g slice 4,
e1 / ep 8, tables 2, partial tiles 8, vectors ~1 (2 CTAs/SM by smem).
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from mogan_kda.utils import prepare_chunk_indices

# Same warp policy for every GEMM that writes an accumulator fragment .
_POLICY = GemmWarpPolicy.FullRow


def _cap(d, cap):
    """min(d, cap): identity for every in-range exponent, only protects 0 x inf on structurally zero / padded entries.
    T.min (not T.if_then_else): a select on per-element values defeats TileLang's auto-vectorisation (probe, GH200)."""
    return T.min(d, cap)


def _f32(x, cast):
    """fp32 view of a possibly bf16 partial element (identity when the partials are fp32)."""
    return T.Cast("float32", x) if cast else x


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_bwd_intra_kda(
    H,
    Hg,
    DK,
    chunk_size,
    scale,
    accum_dtype,
    qk_dtype,
    da_dtype,
    seqlen_dtype,
    is_varlen,
    has_partial,
    split,
    async_mode,
    transpose_gemm,
    p_dtype=None,
    out_dtype=None,
    stage_partials=False,
):
    """``p_dtype``: dtype of dq_p / dk_p (bf16 as written by K3b); ``out_dtype``: dtype of dq / dk (bf16);
    ``stage_partials``: the pass's dq_p / dk_p / dg_p tiles are copied to smem with cp.async at the top of the pass and
    only awaited before the output loop (latency hidden behind the GEMMs)."""
    p_dtype = accum_dtype if p_dtype is None else p_dtype
    out_dtype = accum_dtype if out_dtype is None else out_dtype
    p_bf16 = str(p_dtype).replace("torch.", "") != "float32"
    stage = stage_partials and has_partial
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    SUB = 16
    NSUB = block_S // SUB  # 4 row blocks
    KQ = 16  # channels per pass (= SUB: column block (J, c) of a B tile is 16 wide)
    NP = DK // KQ  # passes

    q_shape = (data_batch_size, num_tokens, Hg, DK)
    g_shape = (data_batch_size, num_tokens, H, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    # optional partial inputs: own dynamic dims (1-element dummies when has_partial is False)
    p_shape = (T.dynamic("part_batch"), T.dynamic("part_tokens"), H, DK)

    # ---- GEMM helpers (T.macro: the eager parser rejects nested defs; buffers are passed as arguments) ----
    # mmc: clear accumulator, mm: accumulate; *t: A read transposed (transpose_A=True)
    @T.macro
    def mmc(a_buf, b_buf, c_buf):
        if async_mode:
            T.wgmma_gemm(a_buf, b_buf, c_buf, clear_accum=True, policy=_POLICY)
        else:
            T.gemm(a_buf, b_buf, c_buf, clear_accum=True, policy=_POLICY)

    @T.macro
    def mm(a_buf, b_buf, c_buf):
        if async_mode:
            T.wgmma_gemm(a_buf, b_buf, c_buf, clear_accum=False, policy=_POLICY)
        else:
            T.gemm(a_buf, b_buf, c_buf, clear_accum=False, policy=_POLICY)

    @T.macro
    def mmct(a_buf, b_buf, c_buf):
        if async_mode:
            T.wgmma_gemm(a_buf, b_buf, c_buf, transpose_A=True, clear_accum=True, policy=_POLICY)
        else:
            T.gemm(a_buf, b_buf, c_buf, transpose_A=True, clear_accum=True, policy=_POLICY)

    @T.macro
    def mmt(a_buf, b_buf, c_buf):
        if async_mode:
            T.wgmma_gemm(a_buf, b_buf, c_buf, transpose_A=True, clear_accum=False, policy=_POLICY)
        else:
            T.gemm(a_buf, b_buf, c_buf, transpose_A=True, clear_accum=False, policy=_POLICY)

    # ---- pass-input loads (issue only; the caller waits): full chunk = cp.async, tail chunk = masked zero-filled stores ----
    @T.macro
    def load_qkg(bb, bh, bhg, left, right, seq_end_idx, c0, q, k, g, q_dst, k_dst, g_dst):
        if right <= seq_end_idx:
            T.async_copy(q[bb, left:right, bhg, c0 : c0 + KQ], q_dst)
            T.async_copy(k[bb, left:right, bhg, c0 : c0 + KQ], k_dst)
            T.async_copy(g[bb, left:right, bh, c0 : c0 + KQ], g_dst)
        else:
            for j_s, j_c in T.Parallel(block_S, KQ):
                if left + j_s < seq_end_idx:
                    q_dst[j_s, j_c] = q[bb, left + j_s, bhg, c0 + j_c]
                    k_dst[j_s, j_c] = k[bb, left + j_s, bhg, c0 + j_c]
                    g_dst[j_s, j_c] = g[bb, left + j_s, bh, c0 + j_c]
                else:
                    q_dst[j_s, j_c] = 0
                    k_dst[j_s, j_c] = 0
                    g_dst[j_s, j_c] = 0

    @T.macro
    def load_part(bb, bh, left, right, seq_end_idx, c0, dq_p, dk_p, dg_p, pq_dst, pk_dst, pg_dst):
        if right <= seq_end_idx:
            T.async_copy(dq_p[bb, left:right, bh, c0 : c0 + KQ], pq_dst)
            T.async_copy(dk_p[bb, left:right, bh, c0 : c0 + KQ], pk_dst)
            T.async_copy(dg_p[bb, left:right, bh, c0 : c0 + KQ], pg_dst)
        else:
            for j_s, j_c in T.Parallel(block_S, KQ):
                if left + j_s < seq_end_idx:
                    pq_dst[j_s, j_c] = dq_p[bb, left + j_s, bh, c0 + j_c]
                    pk_dst[j_s, j_c] = dk_p[bb, left + j_s, bh, c0 + j_c]
                    pg_dst[j_s, j_c] = dg_p[bb, left + j_s, bh, c0 + j_c]
                else:
                    pq_dst[j_s, j_c] = 0
                    pk_dst[j_s, j_c] = 0
                    pg_dst[j_s, j_c] = 0

    @T.macro
    def kernel_body(
        bb,
        bh,
        bhg,
        seq_start_idx,
        chunk_idx,
        seq_end_idx,
        q,
        k,
        g,
        beta,
        dAqk,
        dL,
        dq_p,
        dk_p,
        dg_p,
        dq,
        dk,
        dg,
        dbeta,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        wq_s = T.alloc_shared((block_S, block_S), dtype=da_dtype)  # dAqk strictly lower, [i, j]
        dl_s = T.alloc_shared((block_S, block_S), dtype=da_dtype)  # dL strictly lower, [i, j]
        if not transpose_gemm:
            wqt_s = T.alloc_shared((block_S, block_S), dtype=da_dtype)  # dAqk^T: [j, i], i > j
            dlt_s = T.alloc_shared((block_S, block_S), dtype=da_dtype)  # dL^T: [j, i], i > j
        dd_s = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")  # diagonal of dAqk
        b_s = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")
        db_s = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")  # dbeta accumulator over the passes

        # two hi/lo B-tile pairs: pair 0 = row operands, then beta*k operands; pair 1 = q operands
        bhi0 = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
        blo0 = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
        bhi1 = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
        blo1 = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
        stg = T.alloc_shared((block_S, block_S), dtype=accum_dtype)  # fp32 staging of a GEMM result
        rq_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)  # sum_j dAqk_ij k_j E_ij
        t_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)  # T_i = sum_j dL_ij k_j E_ij
        c_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)  # column term of dk (scale and beta already inside)

        q_s = T.alloc_shared((block_S, KQ), dtype=qk_dtype)
        k_s = T.alloc_shared((block_S, KQ), dtype=qk_dtype)
        g_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)
        e1_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)  # 2^(g_i - g_a(b(i))) <= 1
        ep_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)  # 2^(g_a(b(j)) - g_j) in [1, 2^127]
        tab_s = T.alloc_shared((NSUB, NSUB, KQ), dtype=accum_dtype)  # tab[I, J, c] = 2^(g_aI - g_aJ) (I >= J), else 0
        if stage:  # step b3: this pass's partial tiles
            pq_s = T.alloc_shared((block_S, KQ), dtype=p_dtype)
            pk_s = T.alloc_shared((block_S, KQ), dtype=p_dtype)
            pg_s = T.alloc_shared((block_S, KQ), dtype=accum_dtype)

        f_rq = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        f_rt = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        f_c = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        kt_frag = T.alloc_fragment((block_S, KQ), dtype=accum_dtype)  # k * T, reduced into dbeta
        db_tmp = T.alloc_fragment((block_S,), dtype=accum_dtype)

        # ---- per-chunk operands: beta, dAqk diagonal, dA tiles (strictly lower; the mask is an index predicate) ----
        for j_s in T.Parallel(block_S):
            if left + j_s < seq_end_idx:
                b_s[j_s] = beta[bb, left + j_s, bh]
                dd_s[j_s] = dAqk[bb, left + j_s, bh, j_s]
            else:
                b_s[j_s] = 0
                dd_s[j_s] = 0
            db_s[j_s] = 0
        for j_s, j_t in T.Parallel(block_S, block_S):
            if (left + j_s < seq_end_idx) and (j_t < j_s):
                wq_s[j_s, j_t] = dAqk[bb, left + j_s, bh, j_t]
                dl_s[j_s, j_t] = dL[bb, left + j_s, bh, j_t]
            else:
                wq_s[j_s, j_t] = 0
                dl_s[j_s, j_t] = 0
        if not transpose_gemm:
            # transposed copies [j, i] built from the ROW-MAJOR SHARED tiles (contiguous global reads above; the old
            # strided global gathers were the scalar 16-bit traffic); the masks are already applied
            for j_t, j_s in T.Parallel(block_S, block_S):
                wqt_s[j_t, j_s] = wq_s[j_s, j_t]
                dlt_s[j_t, j_s] = dl_s[j_s, j_t]
        # B tiles: static zero structure (row r is live only in column block r // 16), zero filled once
        for j_s, j_t in T.Parallel(block_S, block_S):
            bhi0[j_s, j_t] = 0
            blo0[j_s, j_t] = 0
            bhi1[j_s, j_t] = 0
            blo1[j_s, j_t] = 0

        for i_p in T.serial(NP):
            c0 = i_p * KQ
            load_qkg(bb, bh, bhg, left, right, seq_end_idx, c0, q, k, g, q_s, k_s, g_s)
            if right <= seq_end_idx:
                T.ptx_wait_group(0)
            if stage:  # b3: partial tiles of this pass, awaited right before the output loop
                T.sync_threads()  # WAR: pass i - 1's output loop is done with pq_s / pk_s / pg_s
                load_part(bb, bh, left, right, seq_end_idx, c0, dq_p, dk_p, dg_p, pq_s, pk_s, pg_s)
                T.ptx_commit_group()

            # ---- the per-element exponent factors, ONCE per pass (2 exp2 per element) ----
            # e1: builds of the column operands and the row combine; ep: row operand build and the column combine.
            for r, c in T.Parallel(block_S, KQ):
                e1_s[r, c] = T.exp2(_cap(g_s[r, c] - g_s[(r // SUB) * SUB, c], 0.0))
                ep_s[r, c] = T.exp2(_cap(g_s[(r // SUB) * SUB, c] - g_s[r, c], 127.0))
            # block-pair table for the combines (256 elements): 2^(g_aI - g_aJ) <= 1 for I >= J, exact 0 elsewhere
            for i_b, j_b, c in T.Parallel(NSUB, NSUB, KQ):
                if i_b >= j_b:
                    tab_s[i_b, j_b, c] = T.exp2(_cap(g_s[SUB * i_b, c] - g_s[SUB * j_b, c], 0.0))
                else:
                    tab_s[i_b, j_b, c] = 0

            # ---- (1) row operands B'[j, (J, c)] = k_j ep_j -> pair 0; issue the row GEMMs ----
            if split:
                for r, c in T.Parallel(block_S, KQ):
                    x = k_s[r, c] * ep_s[r, c]
                    bhi0[r, (r // SUB) * SUB + c] = x
                    blo0[r, (r // SUB) * SUB + c] = x - bhi0[r, (r // SUB) * SUB + c]
            else:
                for r, c in T.Parallel(block_S, KQ):
                    bhi0[r, (r // SUB) * SUB + c] = k_s[r, c] * ep_s[r, c]
            T.fence_proxy_async()
            mmc(wq_s, bhi0, f_rq)
            mmc(dl_s, bhi0, f_rt)
            if split:
                mm(wq_s, blo0, f_rq)
                mm(dl_s, blo0, f_rt)

            # ---- (2) column q operands B''[i, (I, c)] = scale q_i e1_i -> pair 1 (built while the row GEMMs run) ----
            if split:
                for r, c in T.Parallel(block_S, KQ):
                    x = scale * q_s[r, c] * e1_s[r, c]
                    bhi1[r, (r // SUB) * SUB + c] = x
                    blo1[r, (r // SUB) * SUB + c] = x - bhi1[r, (r // SUB) * SUB + c]
            else:
                for r, c in T.Parallel(block_S, KQ):
                    bhi1[r, (r // SUB) * SUB + c] = scale * q_s[r, c] * e1_s[r, c]
            T.fence_proxy_async()
            if transpose_gemm:
                mmct(wq_s, bhi1, f_c)
                if split:
                    mmt(wq_s, blo1, f_c)
            else:
                mmc(wqt_s, bhi1, f_c)
                if split:
                    mm(wqt_s, blo1, f_c)
            # the row GEMMs (older groups) are done, the q GEMMs (2 with split, 1 without) may still be in flight
            if async_mode:
                T.wait_wgmma(2 if split else 1)

            # ---- (3) beta*k operands -> pair 0 (free now); issue the last column GEMMs ----
            if split:
                for r, c in T.Parallel(block_S, KQ):
                    x = b_s[r] * k_s[r, c] * e1_s[r, c]
                    bhi0[r, (r // SUB) * SUB + c] = x
                    blo0[r, (r // SUB) * SUB + c] = x - bhi0[r, (r // SUB) * SUB + c]
            else:
                for r, c in T.Parallel(block_S, KQ):
                    bhi0[r, (r // SUB) * SUB + c] = b_s[r] * k_s[r, c] * e1_s[r, c]
            T.fence_proxy_async()
            if transpose_gemm:
                mmt(dl_s, bhi0, f_c)
                if split:
                    mmt(dl_s, blo0, f_c)
            else:
                mm(dlt_s, bhi0, f_c)
                if split:
                    mm(dlt_s, blo0, f_c)

            # ---- row combines (f_rq / f_rt are complete) while the column GEMMs run; no exp2: e1_i * tab[b(i), J] ----
            T.copy(f_rq, stg)
            for j_s, c in T.Parallel(block_S, KQ):
                rq_s[j_s, c] = e1_s[j_s, c] * (
                    stg[j_s, c] * tab_s[j_s // SUB, 0, c]
                    + stg[j_s, SUB + c] * tab_s[j_s // SUB, 1, c]
                    + stg[j_s, 2 * SUB + c] * tab_s[j_s // SUB, 2, c]
                    + stg[j_s, 3 * SUB + c] * tab_s[j_s // SUB, 3, c]
                )
            T.copy(f_rt, stg)
            for j_s, c in T.Parallel(block_S, KQ):
                t_s[j_s, c] = e1_s[j_s, c] * (
                    stg[j_s, c] * tab_s[j_s // SUB, 0, c]
                    + stg[j_s, SUB + c] * tab_s[j_s // SUB, 1, c]
                    + stg[j_s, 2 * SUB + c] * tab_s[j_s // SUB, 2, c]
                    + stg[j_s, 3 * SUB + c] * tab_s[j_s // SUB, 3, c]
                )
            if async_mode:
                T.wait_wgmma(0)

            # ---- column combine: 2^(g_aI - g_j) = tab[I, b(j)] * ep_j (entries with I < b(j) are exact zeros of stg) ----
            T.copy(f_c, stg)
            for j_s, c in T.Parallel(block_S, KQ):
                c_s[j_s, c] = ep_s[j_s, c] * (
                    stg[j_s, c] * tab_s[0, j_s // SUB, c]
                    + stg[j_s, SUB + c] * tab_s[1, j_s // SUB, c]
                    + stg[j_s, 2 * SUB + c] * tab_s[2, j_s // SUB, c]
                    + stg[j_s, 3 * SUB + c] * tab_s[3, j_s // SUB, c]
                )

            # ---------------- outputs ----------------
            if stage:
                T.ptx_wait_group(0)
                T.sync_threads()
                for j_s, c in T.Parallel(block_S, KQ):
                    kt_frag[j_s, c] = k_s[j_s, c] * t_s[j_s, c]  # rows past the end: k = 0 -> 0
                    if left + j_s < seq_end_idx:
                        dq[bb, left + j_s, bh, c0 + c] = (
                            _f32(pq_s[j_s, c], p_bf16)
                            + scale * rq_s[j_s, c]
                            + scale * dd_s[j_s] * k_s[j_s, c]
                        )
                        dk[bb, left + j_s, bh, c0 + c] = (
                            _f32(pk_s[j_s, c], p_bf16)
                            + b_s[j_s] * t_s[j_s, c]
                            + c_s[j_s, c]
                            + scale * dd_s[j_s] * q_s[j_s, c]
                        )
                        dg[bb, left + j_s, bh, c0 + c] = (
                            pg_s[j_s, c]
                            + q_s[j_s, c] * scale * rq_s[j_s, c]
                            + k_s[j_s, c] * b_s[j_s] * t_s[j_s, c]
                            - k_s[j_s, c] * c_s[j_s, c]
                        )
            else:
                for j_s, c in T.Parallel(block_S, KQ):
                    kt_frag[j_s, c] = k_s[j_s, c] * t_s[j_s, c]  # rows past the end: k = 0 -> 0
                    if left + j_s < seq_end_idx:
                        dq[bb, left + j_s, bh, c0 + c] = (
                            (_f32(dq_p[bb, left + j_s, bh, c0 + c], p_bf16) if has_partial else 0)
                            + scale * rq_s[j_s, c]
                            + scale * dd_s[j_s] * k_s[j_s, c]
                        )
                        dk[bb, left + j_s, bh, c0 + c] = (
                            (_f32(dk_p[bb, left + j_s, bh, c0 + c], p_bf16) if has_partial else 0)
                            + b_s[j_s] * t_s[j_s, c]
                            + c_s[j_s, c]
                            + scale * dd_s[j_s] * q_s[j_s, c]
                        )
                        dg[bb, left + j_s, bh, c0 + c] = (
                            (dg_p[bb, left + j_s, bh, c0 + c] if has_partial else 0)
                            + q_s[j_s, c] * scale * rq_s[j_s, c]
                            + k_s[j_s, c] * b_s[j_s] * t_s[j_s, c]
                            - k_s[j_s, c] * c_s[j_s, c]
                        )
            T.reduce_sum(kt_frag, db_tmp, dim=1, clear=True)
            for j_s in T.Parallel(block_S):
                db_s[j_s] = db_s[j_s] + db_tmp[j_s]

        for j_s in T.Parallel(block_S):
            if left + j_s < seq_end_idx:
                dbeta[bb, left + j_s, bh] = db_s[j_s]

    if is_varlen:

        @T.prim_func
        def tilelang_bwd_intra_kda_kernel(
            q: T.Tensor(q_shape, dtype=qk_dtype),
            k: T.Tensor(q_shape, dtype=qk_dtype),
            g: T.Tensor(g_shape, dtype=accum_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            dAqk: T.Tensor(a_shape, dtype=da_dtype),
            dL: T.Tensor(a_shape, dtype=da_dtype),
            dq_p: T.Tensor(p_shape, dtype=p_dtype),
            dk_p: T.Tensor(p_shape, dtype=p_dtype),
            dg_p: T.Tensor(p_shape, dtype=accum_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            dq: T.Tensor(g_shape, dtype=out_dtype),
            dk: T.Tensor(g_shape, dtype=out_dtype),
            dg: T.Tensor(g_shape, dtype=accum_dtype),
            dbeta: T.Tensor(b_shape, dtype=accum_dtype),
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
                    bb, bh, bhg, seq_start_idx, chunk_idx, seq_end_idx,
                    q, k, g, beta, dAqk, dL, dq_p, dk_p, dg_p, dq, dk, dg, dbeta,
                )

    else:

        @T.prim_func
        def tilelang_bwd_intra_kda_kernel(
            q: T.Tensor(q_shape, dtype=qk_dtype),
            k: T.Tensor(q_shape, dtype=qk_dtype),
            g: T.Tensor(g_shape, dtype=accum_dtype),
            beta: T.Tensor(b_shape, dtype=accum_dtype),
            dAqk: T.Tensor(a_shape, dtype=da_dtype),
            dL: T.Tensor(a_shape, dtype=da_dtype),
            dq_p: T.Tensor(p_shape, dtype=p_dtype),
            dk_p: T.Tensor(p_shape, dtype=p_dtype),
            dg_p: T.Tensor(p_shape, dtype=accum_dtype),
            dq: T.Tensor(g_shape, dtype=out_dtype),
            dk: T.Tensor(g_shape, dtype=out_dtype),
            dg: T.Tensor(g_shape, dtype=accum_dtype),
            dbeta: T.Tensor(b_shape, dtype=accum_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                bb = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                # int32 locals evaluated once (a plain expression would be re-inlined, 64-bit mod / div by the dynamic
                # data_batch_size included, into every global address)
                bb = bc % data_batch_size
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb, bh, bhg, seq_start_idx, chunk_idx, seq_end_idx,
                    q, k, g, beta, dAqk, dL, dq_p, dk_p, dg_p, dq, dk, dg, dbeta,
                )

    return tilelang_bwd_intra_kda_kernel


def bwd_intra_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dL: torch.Tensor,
    dq_p: Optional[torch.Tensor] = None,
    dk_p: Optional[torch.Tensor] = None,
    dg_p: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    """Returns ``(dq, dk, dg, dbeta_g)``: bf16 dq, dk, fp32 dg [B, T, HV, 128] (dg is dg', before the reverse cumsum) and
    fp32 dbeta_g [B, T, HV], each = partial input (if given) + the intra-chunk contribution (see the module docstring).
    ``dq_p`` / ``dk_p`` may be bf16 or fp32 (same dtype for both), ``dg_p`` is fp32."""
    batch_size, num_tokens, Hg, K = q.shape
    H = beta.shape[2]
    assert K == 128 and chunk_size == 64
    assert q.dtype == k.dtype == torch.bfloat16 and k.shape == q.shape and H % Hg == 0
    assert g.dtype == torch.float32 and g.shape == (batch_size, num_tokens, H, K) and g.is_contiguous()
    assert beta.dtype == torch.float32 and beta.is_contiguous()
    assert dAqk.dtype == dL.dtype == torch.bfloat16 and dAqk.shape == (batch_size, num_tokens, H, chunk_size)
    assert dAqk.is_contiguous() and dL.is_contiguous() and q.is_contiguous() and k.is_contiguous()
    has_partial = dq_p is not None
    assert (dq_p is None) == (dk_p is None) == (dg_p is None)
    scale = K ** -0.5 if scale is None else scale

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    if has_partial:
        assert dq_p.dtype == dk_p.dtype in (torch.float32, torch.bfloat16) and dg_p.dtype == torch.float32
        for t in (dq_p, dk_p, dg_p):
            assert t.shape == g.shape and t.is_contiguous()
    else:  # 1-element dummies, never read
        dq_p = dk_p = dg_p = torch.empty((1, 1, H, K), dtype=torch.float32, device=q.device)
    dg = torch.empty_like(g)
    dq, dk = (torch.empty_like(g, dtype=torch.bfloat16) for _ in range(2))
    dbeta = torch.empty_like(beta)

    kernel = tilelang_bwd_intra_kda(
        H,
        Hg,
        K,
        chunk_size,
        float(scale),
        accum_dtype="float32",
        qk_dtype=q.dtype,
        da_dtype=dAqk.dtype,
        seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen,
        has_partial=has_partial,
        split=True,
        async_mode=True,
        transpose_gemm=True,
        p_dtype=dq_p.dtype,
        out_dtype=dq.dtype,
        stage_partials=True,
    )
    if is_varlen:
        kernel(q, k, g, beta, dAqk, dL, dq_p, dk_p, dg_p, cu_seqlens, chunk_indices, dq, dk, dg, dbeta)
    else:
        kernel(q, k, g, beta, dAqk, dL, dq_p, dk_p, dg_p, dq, dk, dg, dbeta, num_chunks)
    return dq, dk, dg, dbeta
