# Derived from mogan_kda/ops/gated_delta_rule/chunk/hopper/fused_fwd.py
# (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT License).
"""Fused forward of the per-channel-gate (KDA) chunk delta rule, Hopper.

Per chunk (64 tokens) and v-head, with the gate-scaled operands prepared by ``kkt_solve_kda(emit_scaled=True)``
(qt = Q exp2(g), kt = K exp2(g), kb = K exp2(g_L - g), gl = exp2(g_L); g = chunk-local log2 cumsum, all factors <= 1)
and Akk = (I + StrictLower(diag(beta) Gamma))^-1, Aqk (lower incl. diagonal, ``scale`` and gate already inside)::

    W  = beta (.) (V - kt S)                (rows scaled by beta; S bf16 copy in smem as GEMM operand)
    Vd = Akk W                              (bf16)
    O  = scale * (qt S) + Aqk Vd
    S <- diag(gl) S + kb^T Vd               (row scaling along K; column scaling in the state_v_first layout)

Grid ``ceil(DV / 64) * num_seqs * HV`` (block_DV = 64), 512 threads:
    WG0 (tx <  128)  state S (fp32 fragment [DK, 64] / [64, DK]); h_shared copy; S <- diag(gl) S + kb^T Vd
    WG1 (tx <  256)  U = kt S, W, Vd = Akk W
    WG2 (tx <  384)  O = scale qt S + Aqk Vd
    producer warps (tx >= 384): P_A (384..415) qt/kt/kb/gl, P_B (416..447) v/beta, P_C (448..479) Akk/Aqk
                                P_S (480..511) store O / h (store_h)
All per-v-head inputs are indexed by bh (q/k GVA expansion already happened in kkt_solve_kda).

Barriers. Every barrier completes once per iteration i_s; consumers wait with parity ``i_s % 2``.
A "thread group" arrives with every one of its threads (arrive_count = number of threads).
  data_is_ready[s]  (s = i_s % 2, count 96)  arrive: P_A, P_B, P_C (3 x 32) after their loads of stage s
                    wait: WG0, WG1, WG2, parity (i_s // 2) % 2.  TMA loads carry the transaction bytes on it.
  data_is_free[s]   (count 384)              arrive: WG0 (after the S-update GEMM), WG1 (after the Vd GEMM), WG2
                    (after the O GEMM) = last use of the stage-s buffers (kt/kb/qt/v/Akk/Aqk/gl/beta).
                    wait: P_A, P_B, P_C before refilling stage s, parity (i_s // 2 + 1) % 2 (first two fills pass).
  bar_start         (count 416)              arrive: WG0, WG1, WG2 (384) right after data_is_ready, and P_S (32) at the
                    top of its round. wait: everyone. It is the start-of-iteration fence: when it completes, every
                    party has finished iteration i_s - 1 completely (h_shared, vd_shared, o_shared readers done).
  bar_s             (count 128)              arrive: WG0 after h_shared <- S (bf16) + fence.
                    wait: WG1, WG2, P_S (h_shared readers). WG0 itself needs no wait.
  bar_vd            (count 128)              arrive: WG1 after vd_shared <- Vd + fence.
                    wait: WG0 (S update), WG2 (O += Aqk Vd).
  bar_ofree         (count 32)               arrive: P_S after it issued the store of O(i_s - 1) (o_shared read).
                    wait: WG2 before o_shared <- O(i_s).
  bar_o             (count 128)              arrive: WG2 once after the loop. wait: P_S before the last masked O store.
Hazard argument (all buffers are single-buffered except the stage inputs):
  h_shared   written by WG0 after bar_start(i); read by WG1/WG2/P_S after bar_s(i); the next write happens after
             bar_start(i+1), which needs WG1, WG2 and P_S to have finished iteration i.
  vd_shared  written by WG1 after bar_s(i); read by WG0/WG2 after bar_vd(i); next write after bar_start(i+1).
  o_shared   written by WG2 after bar_ofree(i); read by P_S in round i+1 after bar_start(i+1) (WG2 arrives there
             after writing it).
  Wait graph per iteration: bar_start <- {all} ; bar_s <- WG0 <- bar_start ; bar_vd <- WG1 <- bar_s ;
  bar_ofree <- P_S <- bar_start. No cycles.
  One phase of every barrier is in flight at a time: a party can only arrive on a barrier for iteration i+1 after
  passing bar_start(i+1), which requires all waiters of iteration i to have arrived at it (i.e. left their waits).

CP (is_cp) is not supported yet; the slots ``cp_seq_map`` / ``raw_cu_seqlens`` and the ``raw_seq_end_idx`` logic are
kept as in the GDN kernel where the intra-card CP hook would go (state init from a preprocessed h0 and the
final-state store on the last piece of a raw sequence).
"""

import torch
import tilelang
import tilelang.language as T

from mogan_kda.utils import prepare_chunk_offsets

BLOCK_DV = 64
# Named barrier for intra-WG1 synchronisation (cp_fwd.py uses id 3 the same way).
BAR_WG1 = 3


@tilelang.jit(
    # out_idx=[-3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_kda_fwd(
    H,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_o,
    is_varlen,
    is_cp,
    state_v_first,
    block_DV=BLOCK_DV,
):
    assert not is_cp, "intra/inter-card CP is not supported for KDA yet"
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    gl_chunks = T.dynamic("gl_chunks")
    raw_batch_size = T.dynamic("raw_batch_size")
    block_S = chunk_size

    lead = 1 if is_varlen else batch_size
    qk_shape = (lead, num_tokens, H, DK)
    v_shape = (lead, num_tokens, H, DV)
    o_shape = (lead, num_tokens, H, DV)
    a_shape = (lead, num_tokens, H, chunk_size)
    b_shape = (lead, num_tokens, H)
    gl_shape = (lead, gl_chunks, H, DK)
    h_shape = (lead, num_chunks, H, DV, DK) if state_v_first else (lead, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DV, DK) if state_v_first else (batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DV, DK) if state_v_first else (raw_batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_kda_fwd_kernel(
        qt: T.Tensor(qk_shape, dtype=qkva_dtype),
        kt: T.Tensor(qk_shape, dtype=qkva_dtype),
        kb: T.Tensor(qk_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        aqk: T.Tensor(a_shape, dtype=qkva_dtype),
        gl: T.Tensor(gl_shape, dtype=accum_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], dtype=seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=512) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H

            DV_start = bv * block_DV
            DV_end = (bv + 1) * block_DV

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            seq_split_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            chunk_split_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            raw_batch_idx = T.alloc_var("int32")
            raw_seq_end_idx = T.alloc_var("int32")
            need_store_final_state = T.alloc_var("bool")
            # CP hook: with is_cp, raw_batch_idx = cp_seq_map[bb] and raw_seq_end_idx = raw_cu_seqlens[raw_batch_idx + 1]
            raw_batch_idx = bb
            raw_seq_end_idx = seq_end_idx
            need_store_final_state = store_final_state & (raw_seq_end_idx == seq_end_idx)

            num_iters = T.alloc_var("int32")
            num_unmasked_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            num_unmasked_iters = (seq_end_idx - seq_start_idx) // block_S

            # ---- shared memory (block_DV = 64: ~178 KB) ----
            qt_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            kt_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            kb_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=qkva_dtype)  # v, then W in place
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)  # Akk
            aq_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)  # Aqk
            gl_shared = T.alloc_shared((2, DK), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            o_shared = T.alloc_shared((block_S, block_DV), dtype=o_dtype)
            h_shared = T.alloc_shared(
                (block_DV, DK) if state_v_first else (DK, block_DV),
                dtype=qkva_dtype,
            )
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)

            h_fragment = T.alloc_fragment(
                (block_DV, DK) if state_v_first else (DK, block_DV),
                dtype=accum_dtype,
            )
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)

            bar_start = T.alloc_barrier(arrive_count=416)
            bar_s = T.alloc_barrier(arrive_count=128)
            bar_vd = T.alloc_barrier(arrive_count=128)
            bar_ofree = T.alloc_barrier(arrive_count=32)
            bar_o = T.alloc_barrier(arrive_count=128)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 40

            if tx < 128:
                # ================= WG0: state S =================
                # Initialize S
                if use_initial_state:
                    if state_v_first:
                        T.copy(h0[bb, bh, DV_start:DV_end, 0:DK], h_fragment)
                    else:
                        T.copy(h0[bb, bh, 0:DK, DV_start:DV_end], h_fragment)
                else:
                    T.clear(h_fragment)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, i_s % 2)

                    # S4[S]: h_shared <- bf16(S), consumed by WG1 (U), WG2 (O) and the store warp
                    T.copy(h_fragment, h_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_s)

                    # S = diag(gl) S (fp32, scaling along K)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            h_fragment[j_v, j_k] *= gl_shared[i_s % 2, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK, block_DV):
                            h_fragment[j_k, j_v] *= gl_shared[i_s % 2, j_k]

                    # S += kb^T Vd
                    T.barrier_wait(bar_vd, i_s % 2)
                    if state_v_first:
                        T.gemm(
                            vd_shared,
                            kb_shared[i_s % 2, :, :],
                            h_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            kb_shared[i_s % 2, :, :],
                            vd_shared,
                            h_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(data_is_free[i_s % 2])

                # Store final S
                if need_store_final_state:
                    if state_v_first:
                        T.copy(h_fragment, ht[raw_batch_idx, bh, DV_start:DV_end, 0:DK])
                    else:
                        T.copy(h_fragment, ht[raw_batch_idx, bh, 0:DK, DV_start:DV_end])

            elif tx < 256:
                # ================= WG1: U, W, Vd =================
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, i_s % 2)

                    # U = kt @ S
                    T.barrier_wait(bar_s, i_s % 2)
                    if state_v_first:
                        T.gemm(
                            kt_shared[i_s % 2, :, :],
                            h_shared,
                            u_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            kt_shared[i_s % 2, :, :],
                            h_shared,
                            u_fragment,
                            clear_accum=True,
                        )

                    # W = beta (V - U), rows; written in place over V (bf16, GEMM operand)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] = (
                            v_shared[i_s % 2, j_s, j_v] - u_fragment[j_s, j_v]
                        ) * b_shared[i_s % 2, j_s]
                    T.sync_threads(BAR_WG1, 128)  # all reads of v_shared done before it is overwritten in place
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]
                    T.fence_proxy_async()
                    T.sync_threads(BAR_WG1, 128)

                    # Vd = Akk @ W
                    T.gemm(
                        a_shared[i_s % 2, :, :],
                        v_shared[i_s % 2, :, :],
                        v_fragment,
                        clear_accum=True,
                    )
                    T.copy(v_fragment, vd_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_vd)

                    T.barrier_arrive(data_is_free[i_s % 2])

            elif tx < 384:
                # ================= WG2: O =================
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, i_s % 2)

                    # O = scale * qt @ S
                    T.barrier_wait(bar_s, i_s % 2)
                    if state_v_first:
                        T.gemm(
                            qt_shared[i_s % 2, :, :],
                            h_shared,
                            o_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            qt_shared[i_s % 2, :, :],
                            h_shared,
                            o_fragment,
                            clear_accum=True,
                        )
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        o_fragment[j_s, j_v] *= scale

                    # O += Aqk @ Vd (Aqk tile straight from smem; scale is already inside Aqk)
                    T.barrier_wait(bar_vd, i_s % 2)
                    T.gemm(aq_shared[i_s % 2, :, :], vd_shared, o_fragment, clear_accum=False)

                    # o_shared <- O once the store warp is done reading O(i_s - 1)
                    T.barrier_wait(bar_ofree, i_s % 2)
                    T.copy(o_fragment, o_shared)

                    T.barrier_arrive(data_is_free[i_s % 2])

                T.barrier_arrive(bar_o)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    # ---- P_A: qt / kt / kb / gl ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                qt[batch_idx, left:right, bh, 0:DK],
                                qt_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                kt[batch_idx, left:right, bh, 0:DK],
                                kt_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                kb[batch_idx, left:right, bh, 0:DK],
                                kb_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            # partial last chunk: zero padded (rows past seq_end belong to the next sequence)
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    qt_shared[i_s % 2, j_s, j_k] = qt[batch_idx, left + j_s, bh, j_k]
                                    kt_shared[i_s % 2, j_s, j_k] = kt[batch_idx, left + j_s, bh, j_k]
                                    kb_shared[i_s % 2, j_s, j_k] = kb[batch_idx, left + j_s, bh, j_k]
                                else:
                                    qt_shared[i_s % 2, j_s, j_k] = 0
                                    kt_shared[i_s % 2, j_s, j_k] = 0
                                    kb_shared[i_s % 2, j_s, j_k] = 0
                            T.fence_proxy_async()

                        # gl of this chunk (always a real chunk; chunk id like h)
                        for j_k in T.Parallel(DK):
                            gl_shared[i_s % 2, j_k] = gl[batch_idx, chunk_start_idx + i_s, bh, j_k]

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 64:
                    # ---- P_B: v / beta ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                v[batch_idx, left:right, bh, DV_start:DV_end],
                                v_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if left + j_s < seq_end_idx:
                                    v_shared[i_s % 2, j_s, j_v] = v[batch_idx, left + j_s, bh, DV_start + j_v]
                                else:
                                    v_shared[i_s % 2, j_s, j_v] = 0
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                                else:
                                    b_shared[i_s % 2, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 96:
                    # ---- P_C: Akk / Aqk ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                aqk[batch_idx, left:right, bh, 0:block_S],
                                aq_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    a_shared[i_s % 2, j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                                    aq_shared[i_s % 2, j_s, j_t] = aqk[batch_idx, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % 2, j_s, j_t] = 0
                                    aq_shared[i_s % 2, j_s, j_t] = 0
                            T.fence_proxy_async()

                        T.barrier_arrive(data_is_ready[i_s % 2])

                else:
                    # ---- P_S: store O (one iteration late) and S ----
                    for i_s in T.serial(num_unmasked_iters):
                        right = seq_start_idx + i_s * block_S
                        left = right - block_S

                        T.barrier_arrive(bar_start)

                        T.barrier_wait(bar_start, i_s % 2)
                        # Store O(i_s - 1)
                        if i_s > 0 and store_o:
                            T.copy(o_shared, o[batch_idx, left:right, bh, DV_start:DV_end])
                        T.barrier_arrive(bar_ofree)

                        T.barrier_wait(bar_s, i_s % 2)
                        # Store S (state entering chunk i_s)
                        if store_h:
                            if state_v_first:
                                T.copy(
                                    h_shared,
                                    h[batch_idx, chunk_start_idx + i_s, bh, DV_start:DV_end, 0:DK],
                                )
                            else:
                                T.copy(
                                    h_shared,
                                    h[batch_idx, chunk_start_idx + i_s, bh, 0:DK, DV_start:DV_end],
                                )

                    if num_unmasked_iters < num_iters:
                        seq_split_idx = seq_start_idx + num_unmasked_iters * block_S
                        chunk_split_idx = chunk_start_idx + num_unmasked_iters
                        right = seq_split_idx
                        left = right - block_S

                        T.barrier_arrive(bar_start)

                        T.barrier_wait(bar_start, num_unmasked_iters % 2)
                        if num_unmasked_iters > 0 and store_o:
                            T.copy(o_shared, o[batch_idx, left:right, bh, DV_start:DV_end])
                        T.barrier_arrive(bar_ofree)

                        T.barrier_wait(bar_s, num_unmasked_iters % 2)
                        if store_h:
                            if state_v_first:
                                T.copy(
                                    h_shared,
                                    h[batch_idx, chunk_split_idx, bh, DV_start:DV_end, 0:DK],
                                )
                            else:
                                T.copy(
                                    h_shared,
                                    h[batch_idx, chunk_split_idx, bh, 0:DK, DV_start:DV_end],
                                )

                    seq_split_idx = seq_start_idx + (num_iters - 1) * block_S

                    # Store the last O (masked), and zero the padding rows after the last sequence
                    T.barrier_wait(bar_o, 0)
                    if store_o and num_iters > 0:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            if seq_split_idx + j_s < seq_end_idx:
                                o[batch_idx, seq_split_idx + j_s, bh, DV_start + j_v] = o_shared[j_s, j_v]
                        if bb == batch_size - 1:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if seq_end_idx + j_s < num_tokens:
                                    o[batch_idx, seq_end_idx + j_s, bh, DV_start + j_v] = 0

    return tilelang_fused_chunk_kda_fwd_kernel


def fused_fwd_kda(
    qt: torch.Tensor,
    kt: torch.Tensor,
    kb: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    aqk: torch.Tensor,
    gl: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cp_seq_map: torch.LongTensor | None = None,
    raw_cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    """Fused KDA forward (see the module docstring). Inputs come from ``kkt_solve_kda(emit_scaled=True)``:
    qt, kt, kb [B, T, HV, 128] bf16, a / aqk [B, T, HV, 64] bf16, gl [B, NC, HV, 128] fp32; v [B, T, HV, 128];
    b [B, T, HV] fp32. ``scale`` multiplies qt S (Aqk already contains it). Returns ``(o, h, final_state)`` with
    h bf16 ``[B, NC, HV, K, V]`` (``[.., V, K]`` if state_v_first) and final_state fp32 ``[N, HV, K, V]``."""
    batch_size, num_tokens, H, K = qt.shape
    V = v.shape[-1]
    chunk_size = a.shape[-1]
    scale = K ** (-0.5) if scale is None else scale
    assert K == V == 128
    assert chunk_size == 64
    assert cp_seq_map is None, "CP is not supported for KDA yet"
    for t in (qt, kt, kb, v, a, aqk, gl, b):
        assert t.is_contiguous()
    output_final_state = output_final_state or False
    output_h = output_h or False
    output_o = output_o if output_o is not None else True

    if cu_seqlens is None:
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_h else 0
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=qt.device)
        chunk_offsets = torch.empty((batch_size + 1), dtype=torch.int32, device=qt.device)
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_h else 0
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    state_shape = (real_batch_size, H, V, K) if state_v_first else (real_batch_size, H, K, V)
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(state_shape, dtype=torch.float32, device=qt.device)
    cp_seq_map = torch.empty((real_batch_size,), dtype=seqlen_dtype, device=qt.device)
    raw_cu_seqlens = torch.empty((real_batch_size + 1,), dtype=seqlen_dtype, device=qt.device)
    h = torch.empty(
        (batch_size, num_chunks, H, V, K) if state_v_first else (batch_size, num_chunks, H, K, V),
        dtype=qt.dtype,
        device=qt.device,
    )
    final_state = torch.empty(state_shape, dtype=torch.float32, device=qt.device)
    o = torch.empty_like(v)

    kernel = tilelang_fused_chunk_kda_fwd(
        H,
        K,
        V,
        chunk_size,
        float(scale),
        qkva_dtype=qt.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        h_dtype=h.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_h=output_h,
        store_o=output_o,
        is_varlen=is_varlen,
        is_cp=False,
        state_v_first=state_v_first,
        block_DV=BLOCK_DV,
    )
    kernel(
        qt, kt, kb, v, a, aqk, gl, b, initial_state, cu_seqlens, chunk_offsets,
        cp_seq_map, raw_cu_seqlens, o, h, final_state,
    )

    return (o if output_o else None), (h if output_h else None), (final_state if output_final_state else None)
