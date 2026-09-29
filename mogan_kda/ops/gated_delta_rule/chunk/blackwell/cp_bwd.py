# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from mogan_kda.utils import prepare_chunk_offsets


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_dh_ws(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    dht_dtype,
    dh_dtype,
    o_dtype,
    seqlen_dtype,
    use_dht,
    store_dh0,
    store_dh,
    is_varlen,
    is_cp,
    state_v_first,
    num_stages=2,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        do_shape = (1, num_tokens, H, DV)
        dh_shape = (
            (1, num_chunks, H, DV, DK)
            if state_v_first
            else (1, num_chunks, H, DK, DV)
        )
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        do_shape = (batch_size, num_tokens, H, DV)
        dh_shape = (
            (batch_size, num_chunks, H, DV, DK)
            if state_v_first
            else (batch_size, num_chunks, H, DK, DV)
        )
    dht_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )
    dh0_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )

    @T.prim_func
    def tilelang_prepare_dh_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        do: T.Tensor(do_shape, dtype=o_dtype),
        dht: T.Tensor(dht_shape, dtype=dht_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, H], dtype=seqlen_dtype),
        dh: T.Tensor(dh_shape, dtype=dh_dtype),
        dh0: T.Tensor(dh0_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.alloc_var("int32")
            num_iters = (
                num_warmup_chunks[bb, bh]
                if is_cp
                else T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            )

            # ===== Staged shared memory (pipeline) =====
            k_shared = T.alloc_shared(
                (num_stages, block_S, DK), dtype=qkva_dtype
            )
            q_shared = T.alloc_shared(
                (num_stages, block_S, DK), dtype=qkva_dtype
            )
            a_shared = T.alloc_shared(
                (num_stages, block_S, block_S), dtype=qkva_dtype
            )
            do_shared = T.alloc_shared(
                (num_stages, block_S, DV), dtype=o_dtype
            )
            g_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )

            # ===== Communication buffers =====
            dh_shared = T.alloc_shared(
                (DV, DK) if state_v_first else (DK, DV),
                dtype=qkva_dtype,
            )
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            # ===== Fragments =====
            dh_fragment_L = T.alloc_fragment(
                (DV // 2, DK) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            dh_fragment_R = T.alloc_fragment(
                (DV // 2, DK) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            xy_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            r_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            g_last_local_dh = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_y = T.alloc_local((1), dtype=accum_dtype)

            # ===== TMEM =====
            dh_tmem_L = T.alloc_tmem(
                (DV // 2, DK) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            dh_tmem_R = T.alloc_tmem(
                (DV // 2, DK) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            r_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
            xy_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)

            # ===== Pipeline barriers =====
            data_is_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_is_free = T.alloc_barrier(arrive_count=[384] * num_stages)

            # ===== tcgen05 barriers =====
            tcbar_0a = T.alloc_barrier(arrive_count=1)  # dh += Rg^T @ dO (L)
            tcbar_0b = T.alloc_barrier(arrive_count=1)  # dh += Rg^T @ dO (R)
            tcbar_1a = T.alloc_barrier(arrive_count=1)  # dh += X^T @ Yg (L)
            tcbar_1b = T.alloc_barrier(arrive_count=1)  # dh += X^T @ Yg (R)
            tcbar_2 = T.alloc_barrier(arrive_count=1)   # P = Q @ K^T
            tcbar_3 = T.alloc_barrier(arrive_count=1)   # R += PL @ X
            tcbar_4 = T.alloc_barrier(arrive_count=1)   # X = Ab @ K
            tcbar_5 = T.alloc_barrier(arrive_count=1)   # Y = K @ dh

            # ===== Compute barriers =====
            # bar_0: full sync (3 consumers 128 each + P3 32)
            bar_0 = T.alloc_barrier(arrive_count=416)
            # bar_1: stage 0 done (Consumer_PR 128 + Consumer_XY 128)
            bar_1 = T.alloc_barrier(arrive_count=256)
            # bar_2: stage 1 done (Consumer_PR 128 + Consumer_XY 128)
            bar_2 = T.alloc_barrier(arrive_count=256)
            # bar_3: stage 2 done (Consumer_dh 128 + Consumer_PR 128)
            bar_3 = T.alloc_barrier(arrive_count=256)
            # bar_4: stage 3 done (Consumer_dh 128 + Consumer_PR 128 + Consumer_XY 128)
            bar_4 = T.alloc_barrier(arrive_count=384)
            # bar_5: stage 4 done (Consumer_dh 128 + Consumer_XY 128)
            bar_5 = T.alloc_barrier(arrive_count=256)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_DH_NREG = 160
            CONSUMER_XY_NREG = 128
            CONSUMER_PR_NREG = 128

            if tx < 128:
                # ============ Consumer_dh ============
                T.set_max_nreg(CONSUMER_DH_NREG, 1)

                if use_dht:
                    if state_v_first:
                        T.copy(dht[bb, bh, :DV // 2, 0:DK], dh_fragment_L)
                        T.copy(dht[bb, bh, DV // 2:, 0:DK], dh_fragment_R)
                    else:
                        T.copy(dht[bb, bh, 0:DK, :DV // 2], dh_fragment_L)
                        T.copy(dht[bb, bh, 0:DK, DV // 2:], dh_fragment_R)
                else:
                    T.clear(dh_fragment_L)
                    T.clear(dh_fragment_R)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0-2] exp(g), copy dh to shared
                    T.barrier_wait(bar_0, i_s % 2)

                    # exp(g)
                    for j_s in T.Parallel(block_S):
                        g_shared[i_s % num_stages, j_s] = T.exp2(
                            g_shared[i_s % num_stages, j_s] * 1.442695
                        )

                    # copy dh to shared (reassemble L/R)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV // 2, DK):
                            dh_shared[j_v, j_k] = dh_fragment_L[j_v, j_k]
                        for j_v, j_k in T.Parallel(DV // 2, DK):
                            dh_shared[j_v + DV // 2, j_k] = dh_fragment_R[j_v, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_shared[j_k, j_v] = dh_fragment_L[j_k, j_v]
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_shared[j_k, j_v + DV // 2] = dh_fragment_R[j_k, j_v]
                    T.fence_proxy_async()

                    T.barrier_arrive(bar_3)

                    # [stage 3] dh decay

                    T.barrier_wait(bar_3, i_s % 2)

                    # dh = g_last * dh  (g_shared already holds exp(g))
                    g_last_local_dh[0] = g_shared[i_s % num_stages, block_S - 1]
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV // 2, DK):
                            dh_fragment_L[j_v, j_k] *= g_last_local_dh[0]
                        for j_v, j_k in T.Parallel(DV // 2, DK):
                            dh_fragment_R[j_v, j_k] *= g_last_local_dh[0]
                    else:
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_fragment_L[j_k, j_v] *= g_last_local_dh[0]
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_fragment_R[j_k, j_v] *= g_last_local_dh[0]

                    T.copy(dh_fragment_L, dh_tmem_L)
                    T.copy(dh_fragment_R, dh_tmem_R)
                    T.barrier_arrive(bar_4)

                    # [stage 4] wait tcgen05 dh += Rg^T @ dO
                    T.barrier_wait(tcbar_0a, i_s % 2)
                    T.barrier_wait(tcbar_0b, i_s % 2)
                    T.copy(dh_tmem_L, dh_fragment_L)
                    T.copy(dh_tmem_R, dh_fragment_R)
                    T.copy(dh_fragment_L, dh_tmem_L)
                    T.copy(dh_fragment_R, dh_tmem_R)
                    T.barrier_arrive(bar_5)

                    # [stage 5] wait tcgen05 dh += X^T @ Yg
                    T.barrier_wait(tcbar_1a, i_s % 2)
                    T.barrier_wait(tcbar_1b, i_s % 2)
                    T.copy(dh_tmem_L, dh_fragment_L)
                    T.copy(dh_tmem_R, dh_fragment_R)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if store_dh0:
                    if state_v_first:
                        T.copy(dh_fragment_L, dh0[bb, bh, :DV // 2, 0:DK])
                        T.copy(dh_fragment_R, dh0[bb, bh, DV // 2:, 0:DK])
                    else:
                        T.copy(dh_fragment_L, dh0[bb, bh, 0:DK, :DV // 2])
                        T.copy(dh_fragment_R, dh0[bb, bh, 0:DK, DV // 2:])

            elif tx < 256:
                # ============ Consumer_PR ============
                T.set_max_nreg(CONSUMER_PR_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0] full sync
                    T.barrier_wait(bar_0, i_s % 2)

                    # Step 1: P = Q @ K^T  (wait tcgen05)
                    T.barrier_wait(tcbar_2, i_s % 2)
                    T.copy(p_tmem, p_fragment)

                    T.barrier_arrive(bar_1)
                    # [stage 1]
                    T.barrier_wait(bar_1, i_s % 2)

                    # Step 2: PL = -Lower(P)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s < j_t:
                            p_fragment[j_s, j_t] = 0
                        else:
                            p_fragment[j_s, j_t] *= -1
                    T.copy(p_fragment, p_shared)
                    T.fence_proxy_async()

                    # Step 3: R = Q + PL @ X  (init tmem with Q, wait tcgen05)
                    T.copy(q_shared[i_s % num_stages, :, :], r_fragment)
                    T.copy(r_fragment, r_tmem)

                    T.barrier_arrive(bar_2)
                    # [stage 2]
                    T.barrier_wait(tcbar_3, i_s % 2)
                    T.copy(r_tmem, r_fragment)

                    T.barrier_arrive(bar_3)
                    # [stage 3]
                    T.barrier_wait(bar_3, i_s % 2)

                    # Step 4: Rg = scale * diag(g) @ R
                    for j_s, j_k in T.Parallel(block_S, DK):
                        r_fragment[j_s, j_k] *= scale * g_shared[i_s % num_stages, j_s]
                    T.copy(r_fragment, q_shared[i_s % num_stages, :, :])
                    T.fence_proxy_async()

                    T.barrier_arrive(bar_4)
                    T.barrier_arrive(data_is_free[i_s % num_stages])

            elif tx < 384:
                # ============ Consumer_XY ============
                T.set_max_nreg(CONSUMER_XY_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0] full sync
                    T.barrier_wait(bar_0, i_s % 2)

                    # Step 1: Ab = A * diag(b)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % num_stages, j_s, j_t] *= b_shared[i_s % num_stages, j_t]
                    T.fence_proxy_async()

                    T.barrier_arrive(bar_1)
                    # [stage 1]
                    # Step 2: X = Ab @ K  (wait tcgen05)
                    T.barrier_wait(tcbar_4, i_s % 2)
                    T.copy(xy_tmem, xy_fragment)
                    T.copy(xy_fragment, x_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_2)

                    # [stage 3]
                    T.barrier_wait(bar_3, i_s % 2)

                    # Step 3: Y = K @ dh  (wait tcgen05)
                    T.barrier_wait(tcbar_5, i_s % 2)
                    T.copy(xy_tmem, xy_fragment)

                    T.barrier_arrive(bar_4)
                    # [stage 4]
                    T.barrier_wait(bar_4, i_s % 2)

                    # Step 4: Yg = -g_last * Y  (g_shared already holds exp(g))
                    g_last_local_y[0] = g_shared[i_s % num_stages, block_S - 1]

                    for j_s, j_v in T.Parallel(block_S, DV):
                        xy_fragment[j_s, j_v] *= -g_last_local_y[0]

                    T.copy(xy_fragment, y_shared)
                    T.fence_proxy_async()

                    T.barrier_arrive(bar_5)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

            else:
                # ============ Producer ============
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    # P0: TMA load K and Q
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                k[batch_idx, left:right, bhg, 0:DK],
                                k_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    k_shared[i_s % num_stages, j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    k_shared[i_s % num_stages, j_s, j_k] = 0
                        if right <= seq_end_idx:
                            T.tma_copy(
                                q[batch_idx, left:right, bhg, 0:DK],
                                q_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    q_shared[i_s % num_stages, j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    q_shared[i_s % num_stages, j_s, j_k] = 0
                            T.fence_proxy_async()

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                elif tx < 384 + 64:
                    # P1: TMA load dO and A
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                do[batch_idx, left:right, bh, 0:DV],
                                do_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_v in T.Parallel(block_S, DV):
                                if left + j_s < seq_end_idx:
                                    do_shared[i_s % num_stages, j_s, j_v] = do[batch_idx, left + j_s, bh, j_v]
                                else:
                                    do_shared[i_s % num_stages, j_s, j_v] = 0
                        if right <= seq_end_idx:
                            T.tma_copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    a_shared[i_s % num_stages, j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % num_stages, j_s, j_t] = 0
                            T.fence_proxy_async()

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                elif tx < 384 + 96:
                    # P2: scalar load g and b (with boundary handling)
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % num_stages, j_s] = g[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]

                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % num_stages, j_s] = b[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % num_stages, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % num_stages, j_s] = 0

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                else:
                    # P3: tcgen05 issuer + store dh
                    for i_s in T.serial(num_iters):
                        T.barrier_arrive(bar_0)

                        # tcgen05: P = Q @ K^T
                        T.barrier_wait(bar_0, i_s % 2)
                        T.tcgen05_gemm(
                            q_shared[i_s % num_stages, :, :],
                            k_shared[i_s % num_stages, :, :],
                            p_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_2,
                            use_2cta=False,
                        )

                        # tcgen05: X = Ab @ K
                        T.barrier_wait(bar_1, i_s % 2)
                        T.tcgen05_gemm(
                            a_shared[i_s % num_stages, :, :],
                            k_shared[i_s % num_stages, :, :],
                            xy_tmem,
                            clear_accum=True,
                            mbar=tcbar_4,
                            use_2cta=False,
                        )

                        # tcgen05: R += PL @ X
                        T.barrier_wait(bar_2, i_s % 2)
                        T.tcgen05_gemm(
                            p_shared,
                            x_shared,
                            r_tmem,
                            clear_accum=False,
                            mbar=tcbar_3,
                            use_2cta=False,
                        )

                        # store dh to global
                        T.barrier_wait(bar_3, i_s % 2)
                        if store_dh:
                            chunk_idx = num_iters - 1 - i_s
                            if state_v_first:
                                T.copy(
                                    dh_shared,
                                    dh[
                                        batch_idx,
                                        chunk_start_idx + chunk_idx,
                                        bh,
                                        0:DV,
                                        0:DK,
                                    ],
                                )
                            else:
                                T.copy(
                                    dh_shared,
                                    dh[
                                        batch_idx,
                                        chunk_start_idx + chunk_idx,
                                        bh,
                                        0:DK,
                                        0:DV,
                                    ],
                                )

                        # tcgen05: Y = K @ dh
                        if state_v_first:
                            T.tcgen05_gemm(
                                k_shared[i_s % num_stages, :, :],
                                dh_shared,
                                xy_tmem,
                                transpose_B=True,
                                clear_accum=True,
                                mbar=tcbar_5,
                                use_2cta=False,
                            )
                        else:
                            T.tcgen05_gemm(
                                k_shared[i_s % num_stages, :, :],
                                dh_shared,
                                xy_tmem,
                                clear_accum=True,
                                mbar=tcbar_5,
                                use_2cta=False,
                            )

                        # tcgen05: dh += Rg^T @ dO
                        T.barrier_wait(bar_4, i_s % 2)
                        if state_v_first:
                            T.tcgen05_gemm(
                                do_shared[i_s % num_stages, :, :DV // 2],
                                q_shared[i_s % num_stages, :, :],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_0a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                do_shared[i_s % num_stages, :, DV // 2:],
                                q_shared[i_s % num_stages, :, :],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_0b,
                                use_2cta=False,
                            )
                        else:
                            T.tcgen05_gemm(
                                q_shared[i_s % num_stages, :, :],
                                do_shared[i_s % num_stages, :, :DV // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_0a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                q_shared[i_s % num_stages, :, :],
                                do_shared[i_s % num_stages, :, DV // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_0b,
                                use_2cta=False,
                            )

                        # tcgen05: dh += X^T @ Yg
                        T.barrier_wait(bar_5, i_s % 2)
                        if state_v_first:
                            T.tcgen05_gemm(
                                y_shared[:, :DV // 2],
                                x_shared,
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_1a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                y_shared[:, DV // 2:],
                                x_shared,
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_1b,
                                use_2cta=False,
                            )
                        else:
                            T.tcgen05_gemm(
                                x_shared,
                                y_shared[:, :DV // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_1a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                x_shared,
                                y_shared[:, DV // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_1b,
                                use_2cta=False,
                            )

    return tilelang_prepare_dh_kernel


def fused_gdr_dh_ws(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    output_dh0: bool = True,
    output_dh: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    num_warmup_chunks: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = do.shape
    chunk_size = a.shape[-1]
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        assert num_warmup_chunks is None
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_dh else 0
        cu_seqlens = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        is_varlen = False
        is_cp = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(
            cu_seqlens, chunk_size
        )
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_dh else 0
        is_varlen = True
        if num_warmup_chunks is None:
            num_warmup_chunks = torch.empty(
                (real_batch_size, H),
                dtype=cu_seqlens.dtype,
                device=k.device,
            )
            is_cp = False
        else:
            is_cp = True

    use_dht = dht is not None
    if dht is None:
        dht = torch.empty(
            (real_batch_size, H, V, K)
            if state_v_first
            else (real_batch_size, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )

    dh = torch.empty(
        (batch_size, num_chunks, H, V, K)
        if state_v_first
        else (batch_size, num_chunks, H, K, V),
        dtype=k.dtype,
        device=k.device,
    )
    dh0 = torch.empty(
        (real_batch_size, H, V, K)
        if state_v_first
        else (real_batch_size, H, K, V),
        dtype=torch.float32,
        device=k.device,
    )

    tilelang_prepare_dh_ws_kernel = tilelang_prepare_dh_ws(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        dht_dtype=dht.dtype,
        dh_dtype=dh.dtype,
        o_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        use_dht=use_dht,
        store_dh0=output_dh0,
        store_dh=output_dh,
        is_varlen=is_varlen,
        is_cp=is_cp,
        state_v_first=state_v_first,
    )
    tilelang_prepare_dh_ws_kernel(
        q, k, a, g, b, do, dht, cu_seqlens, chunk_offsets,
        num_warmup_chunks, dh, dh0,
    )

    if not output_dh0:
        dh0 = None
    if not output_dh:
        dh = None

    return dh, dh0
