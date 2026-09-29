# Derived from mogan_kda/ops/kimi_delta_rule/chunk/hopper/fused_fwd.py and
# mogan_kda/ops/gated_delta_rule/chunk/hopper/fused_bwd.py (Copyright (c) 2026 The Qwen team, Alibaba Group; MIT).
"""K2 of the KDA backward: the reverse state chain (``torch_ref.bwd_dh_torch``).

For every sequence (chunks processed LAST to FIRST) and every v-head, with dS' = gradient of the state leaving the
chunk (fp32 fragment in WG0; the last chunk starts from ``dht`` or 0)::

    dh_c  = dS'                                          (stored, bf16 [lead, NC, HV, K, V])
    dVd   = Aqk^T dO + kbar dS'                          (stored, bf16 [B, T, HV, V]; Aqk already contains ``scale``)
    dS    = diag(gl) dS' + q~^T (scale dO) + (-w)^T dVd  (K5 of the design: -w is passed in, so all GEMMs accumulate)
    dh0   = dS after the first chunk of the sequence     (fp32, only if ``output_dh0``)

Inputs (all per v-head, bf16 unless noted): do [B,T,HV,V]; aqk [B,T,HV,64]; nw = -w, qt, kb [B,T,HV,K] (K0/K1
outputs); gl fp32 [lead,NC,HV,K]; dht fp32 [N,HV,K,V] ([N,HV,V,K] with state_v_first) or None. Rows past a sequence end
are zero padded when loaded (manual loads) and masked on store.

Deviation from the oracle: ``dgl = sum_v S (.) dS'`` is NOT produced here. K3b needs both h and dh anyway, so it
computes dgl there (one CTA covers both V halves, hence no per-half partials and no atomics). dh / dVd / dh0 are
V-half disjoint outputs (grid = 2 V halves x sequences x HV), so no reduction across CTAs exists.
dtypes: dh, dVd bf16 (consumed as GEMM operands in K3, FLA also stores them in the model dtype; the chain itself is
fp32 in registers), dh0 fp32.
scale is applied to dO in fp32 and rounded once to bf16 (``dos``; a bf16 scale constant would carry a systematic
2.8e-3 relative error).

Warp roles (384 threads, block_DV = 64):
    WG0  tx <  128   state gradient F = dS' (fp32 [K, 64] or [64, K]): ds_shared <- bf16(F); F *= gl; F += q~^T dos;
                     F += (-w)^T vd
    WG1  tx <  256   dos = scale dO (bf16, smem); vd = Aqk^T dO + kbar ds (fp32 fragment) -> vd_shared
    P_A  tx <  288   TMA: qt, nw              P_B  tx <  320   TMA: kb; manual: gl
    P_C  tx <  352   TMA: do, aqk             P_S  tx <  384   store dh (ds_shared) and dVd (vd_shared)
Iteration i_s handles chunk c = num_iters - 1 - i_s of the sequence; stage s = i_s % 2 holds the loaded inputs.

Barriers (all complete once per iteration; consumers wait with parity i_s % 2, stage barriers (i_s // 2) % 2):
  data_is_ready[s]  count 96   arrive: P_A, P_B, P_C (3 x 32) after their loads of stage s (TMA bytes ride on it)
                               wait:   WG0, WG1
  data_is_free[s]   count 256  arrive: WG0 (after its last GEMM reading qt / nw / gl), WG1 (after its last GEMM reading
                               kb / do / aqk)   wait: P_A, P_B, P_C before refilling stage s, parity (i_s // 2 + 1) % 2
  bar_start         count 288  arrive: WG0, WG1 (256) right after data_is_ready, P_S (32) at the top of its round
                               wait:   WG0, WG1, P_S. When it completes for iteration i every party has finished
                               iteration i - 1 completely (P_S included: its stores read ds_shared / vd_shared).
                               Protects ds_shared, vd_shared, dos_shared against overwrite.
  bar_s             count 128  arrive: WG0 after ds_shared <- bf16(F) + fence.   wait: WG1 (kbar ds), P_S (dh store)
  bar_vd            count 128  arrive: WG1 after vd_shared, dos_shared written + fence.  wait: WG0 (F GEMMs), P_S
Hazards: ds_shared written by WG0 after bar_start(i), read by WG1 / P_S after bar_s(i), next write after
bar_start(i + 1) (which needs WG1 and P_S done with iteration i). vd_shared / dos_shared written by WG1 after
bar_start(i), read by WG0 (and P_S for vd) after bar_vd(i), next write after bar_start(i + 1). Wait graph per
iteration: bar_start <- {WG0, WG1, P_S}; bar_s <- WG0 <- bar_start; bar_vd <- WG1 <- bar_s; no cycle.

Prefix invariance: when dO = 0 (and dht = 0) from chunk p on, every value derived from those chunks is an exact
0 x finite: dVd0 = Aqk^T dO = 0, dS' = 0, hence dVd = 0, F = gl * 0 + q~^T 0 + (-w)^T 0 = 0; all stage inputs are finite
and partial chunks are zero padded, no uninitialised shared memory is ever read.

Shared memory (KB): qt / nw / kb 3 x 2 x 16 = 96, do 2 x 8, aqk 2 x 8, gl 1, ds 16, vd 8, dos 8 -> 161 (1 CTA/SM).
Registers: F 64 (WG0), vd 32 + dos 32 fragments (WG1), producers 40.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from mogan_kda.utils import prepare_chunk_offsets

BLOCK_DV = 64
_POLICY = GemmWarpPolicy.FullRow


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_bwd_dh_kda(
    H,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    seqlen_dtype,
    use_dht,
    store_dh0,
    is_varlen,
    state_v_first,
    block_DV=BLOCK_DV,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    gl_chunks = T.dynamic("gl_chunks")
    block_S = chunk_size

    lead = 1 if is_varlen else batch_size
    qk_shape = (lead, num_tokens, H, DK)
    v_shape = (lead, num_tokens, H, DV)
    a_shape = (lead, num_tokens, H, chunk_size)
    gl_shape = (lead, gl_chunks, H, DK)
    dh_shape = (lead, num_chunks, H, DV, DK) if state_v_first else (lead, num_chunks, H, DK, DV)
    st_shape = (batch_size, H, DV, DK) if state_v_first else (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_bwd_dh_kda_kernel(
        do: T.Tensor(v_shape, dtype=qkva_dtype),
        aqk: T.Tensor(a_shape, dtype=qkva_dtype),
        nw: T.Tensor(qk_shape, dtype=qkva_dtype),
        qt: T.Tensor(qk_shape, dtype=qkva_dtype),
        kb: T.Tensor(qk_shape, dtype=qkva_dtype),
        gl: T.Tensor(gl_shape, dtype=accum_dtype),
        dht: T.Tensor(st_shape, dtype=accum_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        dh: T.Tensor(dh_shape, dtype=qkva_dtype),
        dvd: T.Tensor(v_shape, dtype=qkva_dtype),
        dh0: T.Tensor(st_shape, dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=384) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H

            DV_start = bv * block_DV
            DV_end = (bv + 1) * block_DV

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            num_iters = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            # ---- shared memory (~161 KB) ----
            qt_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            nw_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            kb_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            do_shared = T.alloc_shared((2, block_S, block_DV), dtype=qkva_dtype)
            aq_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            gl_shared = T.alloc_shared((2, DK), dtype=accum_dtype, scope="shared")

            ds_shared = T.alloc_shared(
                (block_DV, DK) if state_v_first else (DK, block_DV),
                dtype=qkva_dtype,
            )
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            dos_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)

            f_fragment = T.alloc_fragment(
                (block_DV, DK) if state_v_first else (DK, block_DV),
                dtype=accum_dtype,
            )
            vd_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dos_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[256] * 2)

            bar_start = T.alloc_barrier(arrive_count=288)
            bar_s = T.alloc_barrier(arrive_count=128)
            bar_vd = T.alloc_barrier(arrive_count=128)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 40

            if tx < 128:
                # ================= WG0: state gradient F = dS' =================
                if use_dht:
                    if state_v_first:
                        T.copy(dht[bb, bh, DV_start:DV_end, 0:DK], f_fragment)
                    else:
                        T.copy(dht[bb, bh, 0:DK, DV_start:DV_end], f_fragment)
                else:
                    T.clear(f_fragment)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, i_s % 2)

                    # dh_c = bf16(dS'), consumed by WG1 (kbar dS'), the store warp
                    T.copy(f_fragment, ds_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_s)

                    # F = diag(gl) F (scaling along K)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            f_fragment[j_v, j_k] *= gl_shared[i_s % 2, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK, block_DV):
                            f_fragment[j_k, j_v] *= gl_shared[i_s % 2, j_k]

                    # F += q~^T (scale dO) ; F += (-w)^T dVd
                    T.barrier_wait(bar_vd, i_s % 2)
                    if state_v_first:
                        T.gemm(dos_shared, qt_shared[i_s % 2, :, :], f_fragment, transpose_A=True, clear_accum=False)
                        T.gemm(vd_shared, nw_shared[i_s % 2, :, :], f_fragment, transpose_A=True, clear_accum=False)
                    else:
                        T.gemm(qt_shared[i_s % 2, :, :], dos_shared, f_fragment, transpose_A=True, clear_accum=False)
                        T.gemm(nw_shared[i_s % 2, :, :], vd_shared, f_fragment, transpose_A=True, clear_accum=False)
                    T.barrier_arrive(data_is_free[i_s % 2])

                if store_dh0:
                    if state_v_first:
                        T.copy(f_fragment, dh0[bb, bh, DV_start:DV_end, 0:DK])
                    else:
                        T.copy(f_fragment, dh0[bb, bh, 0:DK, DV_start:DV_end])

            elif tx < 256:
                # ================= WG1: dos, dVd =================
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, i_s % 2)

                    # dos = bf16(scale * dO), scaled in fp32
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        dos_fragment[j_s, j_v] = do_shared[i_s % 2, j_s, j_v]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        dos_fragment[j_s, j_v] = dos_fragment[j_s, j_v] * scale
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        dos_shared[j_s, j_v] = dos_fragment[j_s, j_v]

                    # dVd = Aqk^T dO (independent of ds) + kbar ds
                    T.gemm(
                        aq_shared[i_s % 2, :, :],
                        do_shared[i_s % 2, :, :],
                        vd_fragment,
                        transpose_A=True,
                        clear_accum=True,
                        policy=_POLICY,
                    )
                    T.barrier_wait(bar_s, i_s % 2)
                    if state_v_first:
                        T.gemm(
                            kb_shared[i_s % 2, :, :],
                            ds_shared,
                            vd_fragment,
                            transpose_B=True,
                            clear_accum=False,
                            policy=_POLICY,
                        )
                    else:
                        T.gemm(
                            kb_shared[i_s % 2, :, :],
                            ds_shared,
                            vd_fragment,
                            clear_accum=False,
                            policy=_POLICY,
                        )
                    T.copy(vd_fragment, vd_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_vd)

                    T.barrier_arrive(data_is_free[i_s % 2])

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 256 + 32:
                    # ---- P_A: qt / nw ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + (num_iters - 1 - i_s) * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                qt[batch_idx, left:right, bh, 0:DK],
                                qt_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                nw[batch_idx, left:right, bh, 0:DK],
                                nw_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:  # partial chunk: zero padded (rows past seq_end belong to the next sequence)
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    qt_shared[i_s % 2, j_s, j_k] = qt[batch_idx, left + j_s, bh, j_k]
                                    nw_shared[i_s % 2, j_s, j_k] = nw[batch_idx, left + j_s, bh, j_k]
                                else:
                                    qt_shared[i_s % 2, j_s, j_k] = 0
                                    nw_shared[i_s % 2, j_s, j_k] = 0
                            T.fence_proxy_async()

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 256 + 64:
                    # ---- P_B: kb / gl ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + (num_iters - 1 - i_s) * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                kb[batch_idx, left:right, bh, 0:DK],
                                kb_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    kb_shared[i_s % 2, j_s, j_k] = kb[batch_idx, left + j_s, bh, j_k]
                                else:
                                    kb_shared[i_s % 2, j_s, j_k] = 0
                            T.fence_proxy_async()

                        # gl of this chunk (chunk id like h)
                        for j_k in T.Parallel(DK):
                            gl_shared[i_s % 2, j_k] = gl[batch_idx, chunk_start_idx + (num_iters - 1 - i_s), bh, j_k]

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 256 + 96:
                    # ---- P_C: dO half / Aqk ----
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + (num_iters - 1 - i_s) * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                do[batch_idx, left:right, bh, DV_start:DV_end],
                                do_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                aqk[batch_idx, left:right, bh, 0:block_S],
                                aq_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if left + j_s < seq_end_idx:
                                    do_shared[i_s % 2, j_s, j_v] = do[batch_idx, left + j_s, bh, DV_start + j_v]
                                else:
                                    do_shared[i_s % 2, j_s, j_v] = 0
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    aq_shared[i_s % 2, j_s, j_t] = aqk[batch_idx, left + j_s, bh, j_t]
                                else:
                                    aq_shared[i_s % 2, j_s, j_t] = 0
                            T.fence_proxy_async()

                        T.barrier_arrive(data_is_ready[i_s % 2])

                else:
                    # ---- P_S: store dh (state gradient leaving the chunk) and dVd ----
                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - 1 - i_s) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_start)
                        T.barrier_wait(bar_start, i_s % 2)

                        T.barrier_wait(bar_s, i_s % 2)
                        if state_v_first:
                            T.copy(
                                ds_shared,
                                dh[batch_idx, chunk_start_idx + (num_iters - 1 - i_s), bh, DV_start:DV_end, 0:DK],
                            )
                        else:
                            T.copy(
                                ds_shared,
                                dh[batch_idx, chunk_start_idx + (num_iters - 1 - i_s), bh, 0:DK, DV_start:DV_end],
                            )

                        T.barrier_wait(bar_vd, i_s % 2)
                        if right <= seq_end_idx:
                            T.copy(vd_shared, dvd[batch_idx, left:right, bh, DV_start:DV_end])
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if left + j_s < seq_end_idx:
                                    dvd[batch_idx, left + j_s, bh, DV_start + j_v] = vd_shared[j_s, j_v]

    return tilelang_bwd_dh_kda_kernel


def bwd_dh_kda(
    do: torch.Tensor,
    aqk: torch.Tensor,
    nw: torch.Tensor,
    qt: torch.Tensor,
    kb: torch.Tensor,
    gl: torch.Tensor,
    dht: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    state_v_first: bool = False,
    output_dh0: bool = False,
    chunk_size: int = 64,
):
    """K2 (see the module docstring). Returns ``(dh, dvd, dh0)``: dh bf16 [lead, NC, HV, K, V] ([.., V, K] with
    state_v_first), dvd bf16 [B, T, HV, V], dh0 fp32 [N, HV, K, V] (or None). nw = -w from ``bwd_w_kda(negate=True)``."""
    B, Tn, H, V = do.shape
    K = qt.shape[-1]
    assert K == V == 128 and chunk_size == 64
    assert do.dtype == aqk.dtype == nw.dtype == qt.dtype == kb.dtype == torch.bfloat16
    assert gl.dtype == torch.float32
    for t in (do, aqk, nw, qt, kb, gl):
        assert t.is_contiguous()
    scale = K ** -0.5 if scale is None else scale

    if cu_seqlens is None:
        real_batch_size = B
        num_chunks = tilelang.cdiv(Tn, chunk_size)
        cu_seqlens = torch.empty((B + 1), dtype=torch.int32, device=do.device)
        chunk_offsets = torch.empty((B + 1), dtype=torch.int32, device=do.device)
        seqlen_dtype, is_varlen = torch.int32, False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        seqlen_dtype, is_varlen = cu_seqlens.dtype, True

    st_shape = (real_batch_size, H, V, K) if state_v_first else (real_batch_size, H, K, V)
    use_dht = dht is not None
    if dht is None:
        dht = torch.empty(st_shape, dtype=torch.float32, device=do.device)
    else:
        assert dht.dtype == torch.float32 and dht.shape == st_shape and dht.is_contiguous()
    lead = 1 if is_varlen else B
    dh = torch.empty(
        (lead, num_chunks, H, V, K) if state_v_first else (lead, num_chunks, H, K, V),
        dtype=do.dtype, device=do.device,
    )
    dvd = torch.empty_like(do)
    dh0 = torch.empty(st_shape, dtype=torch.float32, device=do.device)

    kernel = tilelang_bwd_dh_kda(
        H, K, V, chunk_size, float(scale),
        accum_dtype="float32", qkva_dtype=do.dtype, seqlen_dtype=seqlen_dtype,
        use_dht=use_dht, store_dh0=output_dh0, is_varlen=is_varlen, state_v_first=state_v_first,
        block_DV=BLOCK_DV,
    )
    kernel(do, aqk, nw, qt, kb, gl, dht, cu_seqlens, chunk_offsets, dh, dvd, dh0)
    return dh, dvd, (dh0 if output_dh0 else None)
