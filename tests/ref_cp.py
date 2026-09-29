# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Plain-torch references for the CP support kernels used by ``test_cp_kernels.py``.

Unlike :mod:`ref_gdr` (an arch-independent float64 *math* oracle), these are literal
transcriptions of each kernel's index arithmetic: floor vs ceil, which token of a chunk
is sampled, which side M multiplies on. They exist to pin the *convention*, so a changed
one fails with a readable diff instead of a slightly-off gradient three layers up. That
makes them change-detectors by design -- the right shape for heuristics that have no
clean math spec, which is exactly why they live here and not in ``ref_gdr``.
"""
import torch

from mogan_kda.ops.gated_delta_rule.chunk import CHUNK_SIZE

THRESHOLD = -10.0          # the kernels' default log-decay cutoff


def ref_warmup(g, cu, ht_mask, *, reverse):
    """Transcription of ``tilelang_get_warmup_chunks_kernel``.

    ``num_iters`` floors (unlike the bidi kernel, which ceils), and where ``ht_mask`` is
    set the kernel writes only ``num_warmup=0`` -- it leaves that row of
    ``fallback_mask`` untouched, so the caller must not read it. Returned as None here to
    make that explicit.
    """
    cu_list = cu.tolist()
    n_cp = len(cu_list) - 1
    num_heads = g.shape[2]
    num_warmup = torch.zeros(n_cp, num_heads, dtype=cu.dtype, device=g.device)
    fallback = torch.zeros(n_cp, num_heads, dtype=torch.bool, device=g.device)
    valid = torch.ones(n_cp, num_heads, dtype=torch.bool, device=g.device)

    for bb in range(n_cp):
        start, end = cu_list[bb], cu_list[bb + 1]
        num_iters = (end - start) // CHUNK_SIZE
        if ht_mask[bb]:
            num_warmup[bb] = 0
            valid[bb] = False
            continue
        num_warmup[bb] = num_iters
        fallback[bb] = True
        cumsum = torch.zeros(num_heads, dtype=torch.float64, device=g.device)
        for i_s in range(num_iters):
            idx = start + (i_s + 1) * CHUNK_SIZE - 1 if reverse else end - i_s * CHUNK_SIZE - 1
            cumsum += g[0, idx].double()
            hit = (cumsum < THRESHOLD) & (num_warmup[bb] == num_iters)
            num_warmup[bb][hit] = i_s + 1
            fallback[bb][hit] = False
    return num_warmup, fallback, valid


def ref_warmup_bidi(
    g, cu, ht_mask_fwd, ht_mask_bwd, seq_map_r2c=None, force_inter_boundaries=False,
):
    """Transcription of ``tilelang_get_warmup_chunks_bidi_kernel``.

    Differences from the single-direction kernel that matter: ``num_iters`` *ceils*, the
    backward sample index is clamped to the last token, ``fallback`` is written on every
    row, and the returned forward warmup count is ``max(n_fwd, n_bwd)`` -- one prepare_h
    launch has to cover both directions' needs.

    ``force_inter_boundaries`` mirrors the kernel's optional override: the partitions of
    the first and last raw sequence (``bb < seq_map_r2c[1]`` or ``bb >= seq_map_r2c[-2]``)
    are forced to full warmup, since inter-card correction owns those card boundaries.
    """
    cu_list = cu.tolist()
    n_cp = len(cu_list) - 1
    H = g.shape[2]
    dev = g.device
    n_fwd = torch.zeros(n_cp, H, dtype=cu.dtype, device=dev)
    n_bwd = torch.zeros(n_cp, H, dtype=cu.dtype, device=dev)
    f_fwd = torch.zeros(n_cp, H, dtype=torch.bool, device=dev)
    f_bwd = torch.zeros(n_cp, H, dtype=torch.bool, device=dev)

    if force_inter_boundaries:
        smap = seq_map_r2c.tolist()
        first_end, last_start = smap[1], smap[-2]

    for bb in range(n_cp):
        start, end = cu_list[bb], cu_list[bb + 1]
        num_iters = -(-(end - start) // CHUNK_SIZE)
        if ht_mask_fwd[bb]:
            n_fwd[bb], f_fwd[bb] = 0, False
        else:
            n_fwd[bb], f_fwd[bb] = num_iters, True
        if ht_mask_bwd[bb]:
            n_bwd[bb], f_bwd[bb] = 0, False
        else:
            n_bwd[bb], f_bwd[bb] = num_iters, True

        cs_fwd = torch.zeros(H, dtype=torch.float64, device=dev)
        cs_bwd = torch.zeros(H, dtype=torch.float64, device=dev)
        for i_s in range(num_iters):
            cs_fwd += g[0, end - i_s * CHUNK_SIZE - 1].double()
            hit = (cs_fwd < THRESHOLD) & (n_fwd[bb] == num_iters)
            n_fwd[bb][hit] = i_s + 1
            f_fwd[bb][hit] = False

            bwd_idx = min(start + (i_s + 1) * CHUNK_SIZE - 1, end - 1)
            cs_bwd += g[0, bwd_idx].double()
            hit = (cs_bwd < THRESHOLD) & (n_bwd[bb] == num_iters)
            n_bwd[bb][hit] = i_s + 1
            f_bwd[bb][hit] = False

        if force_inter_boundaries and (bb < first_end or bb >= last_start):
            n_fwd[bb], n_bwd[bb] = num_iters, num_iters
            f_fwd[bb], f_bwd[bb] = True, True

    return torch.maximum(n_fwd, n_bwd), n_bwd, f_fwd, f_bwd


def ref_scan(
    ht, mt, fallback, seq_map, *,
    raw_h0=None, state_v_first=False, reverse=False, transpose_m=False,
):
    """Transcription of ``tilelang_correct_h0``'s ``scan_body``, all three modes at once.

    Returns ``(cp_h0, h_card, m_card)``:

    * ``cp_h0[idx]`` -- the state *entering* cp segment ``idx`` (correct mode),
    * ``h_card[bb]`` -- the state *leaving* the last segment of raw seq ``bb``
      (aggregate mode),
    * ``m_card[bb]`` -- the ordered M product over the raw sequence's segments. A
      non-fallback segment zeroes the accumulator, and because later segments multiply
      into it, that zero is permanent: M is non-zero only when *every* segment of the
      sequence is a fallback. That is the intended meaning -- if any chunk's history is
      already dead, no correction can transfer across the sequence at all.
    """
    ht = ht.double()
    mt = mt.double()
    K = mt.shape[-1]
    H = ht.shape[1]
    n_raw = seq_map.shape[0] - 1
    seq_map = seq_map.tolist()

    cp_h0 = torch.zeros_like(ht)
    h_card = torch.zeros((n_raw,) + ht.shape[1:], dtype=torch.float64, device=ht.device)
    m_card = torch.zeros((n_raw, H, K, K), dtype=torch.float64, device=ht.device)

    for bb in range(n_raw):
        lo, hi = seq_map[bb], seq_map[bb + 1]
        order = range(hi - 1, lo - 1, -1) if reverse else range(lo, hi)
        for h in range(H):
            state = (
                raw_h0[bb, h].double() if raw_h0 is not None
                else torch.zeros_like(ht[0, h])
            )
            m_run = torch.eye(K, dtype=torch.float64, device=ht.device)
            for idx in order:
                cp_h0[idx, h] = state
                prev, m = state, mt[idx, h]
                state = ht[idx, h].clone()
                if fallback[idx, h]:
                    if state_v_first:
                        state += prev @ (m if transpose_m else m.T)
                    else:
                        state += (m.T if transpose_m else m) @ prev
                    m_run = m @ m_run
                else:
                    m_run = torch.zeros_like(m_run)
            h_card[bb, h] = state
            m_card[bb, h] = m_run
    return cp_h0, h_card, m_card


def rel_max(actual, expected):
    """Max-abs relative error. Distinct from :func:`gdr_common.assert_relative` (an L2
    norm): the scan kernels are checked for an O(1)-wrong index/transpose, which the
    worst single element catches more sharply than a norm that averages it away."""
    diff = (actual.double() - expected.double()).abs().max().item()
    return diff / (expected.double().abs().max().item() + 1e-30)
