# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import math
from typing import Iterator

import tilelang
import torch
import torch.nn.functional as F

from mogan_kda.ops.utils import chunk_local_cumsum
from mogan_kda.utils import l2norm

from mogan_kda.ops.gated_delta_rule.chunk import (  # noqa: F401 -- re-exports
    CHUNK_SIZE,
    correct_initial_states,
    correct_terminal_states,
    fused_gdr_bwd,
    fused_gdr_dh,
    fused_gdr_fwd,
    fused_gdr_h,
    get_warmup_chunks,
    get_warmup_chunks_bidi,
    kkt_solve,
)
from mogan_kda.ops.gated_delta_rule.chunk.cp.context import (
    _build_intra_cp_context,
)
from .utils import (
    MEASURED, ShapeSpec, case_key, lcp_grid, append_cache, cache_key,
    cache_row_complete, read_cache, rewrite_cache,
)

K_DIM = V_DIM = 128

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16, "half": torch.float16}


def torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; "
                         f"expected one of {sorted(_DTYPES)}") from None


def bench(fn, warmup_ms: float = 25, rep_ms: float = 100) -> float:
    return tilelang.profiler.do_bench(fn, warmup=warmup_ms, rep=rep_ms)


def generate_inputs(T: int, Hk: int, Hv: int, dtype: torch.dtype,
                    swa_ratio: float, seed: int = 42) -> dict:
    dev = "cuda"
    torch.manual_seed(seed)
    q = l2norm(torch.randn(1, T, Hk, K_DIM, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, T, Hk, K_DIM, device=dev, dtype=dtype))
    v = torch.randn(1, T, Hv, V_DIM, device=dev, dtype=dtype)
    do = torch.randn(1, T, Hv, V_DIM, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    swa = torch.zeros(Hv, dtype=torch.bool, device=dev)
    swa[: math.ceil(swa_ratio * Hv)] = 1
    swa = swa[torch.randperm(Hv, device=dev)]
    g[:, :, ~swa] = 0.0
    return dict(q=q, k=k, v=v, do=do, g=g, beta=beta, scale=K_DIM ** -0.5)


def measure_shape(spec: ShapeSpec, lcps, *, P: int, chunk: int = CHUNK_SIZE,
                  device: str | None = None, warmup_ms: float = 25,
                  rep_ms: float = 100, seed: int = 42) -> Iterator[dict]:
    dev = "cuda"
    if device is None:
        device = torch.cuda.get_device_name()
    dtype = torch_dtype(spec.dtype)
    cu_raw = torch.tensor(spec.cu, device=dev, dtype=torch.int32)
    chunks = spec.chunks(chunk)

    # inputs / gate cumsum / KKT solve are per-shape (raw cu_seqlens, L_cp-independent)
    inp = generate_inputs(spec.T, spec.Hk, spec.H, dtype, spec.swa_ratio, seed)
    g_c = chunk_local_cumsum(inp["g"], cu_seqlens=cu_raw, chunk_size=chunk)
    A = kkt_solve(inp["k"], inp["beta"], cu_seqlens=cu_raw, chunk_size=chunk)

    avg_per_head = inp["g"].float().mean(dim=1).reshape(-1).tolist()
    gate_avg_str = ":".join(f"{a:.6g}" for a in avg_per_head)

    for lcp in lcps:
        ctx = _build_intra_cp_context(cu_raw, chunk, chunks, int(lcp))
        cp_cu = ctx.intra_cp_cu_seqlens
        n_part = cp_cu.numel() - 1

        nw, fb_mask = get_warmup_chunks(
            g=g_c, cu_seqlens=cp_cu, ht_mask=ctx.ht_mask,
            chunk_size=chunk, threshold=-10.0,
        )
        # Untimed: the bidi scan returns only the two elementwise maxima, so it
        # cannot stand in for the forward-only call above (a step makes both).
        nw_bidi, nw_bwd, _fb_fwd, fb_bwd = get_warmup_chunks_bidi(
            g=g_c, cu_seqlens=cp_cu, ht_mask_fwd=ctx.ht_mask,
            ht_mask_bwd=ctx.ht_mask_bwd, chunk_size=chunk, threshold=-10.0,
        )

        # Untimed setup feeding the timed closures below (fused_bwd needs h_states).
        _, ht, mt = fused_gdr_h(
            k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
            initial_state=None, output_final_state=True, output_h=False,
            cu_seqlens=cp_cu, num_warmup_chunks=nw,
        )
        cp_h0 = correct_initial_states(
            raw_h0=None, ht_buffer=ht, mt_buffer=mt,
            fallback_mask=fb_mask, seq_map_r2c=ctx.seq_map_r2c,
        )
        h_states, _, _ = fused_gdr_h(
            k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
            initial_state=cp_h0, output_final_state=False, output_h=True,
            cu_seqlens=cp_cu,
        )
        _, dht_buffer = fused_gdr_dh(
            q=inp["q"], k=inp["k"], a=A, g=g_c, b=inp["beta"], do=inp["do"],
            dht=None, output_dh0=True, output_dh=False, scale=inp["scale"],
            cu_seqlens=cp_cu, num_warmup_chunks=nw_bwd,
        )

        def run_prepare_h():
            return fused_gdr_h(
                k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                initial_state=None, output_final_state=True, output_h=False,
                cu_seqlens=cp_cu, num_warmup_chunks=nw,
            )

        def run_prepare_h_bidi():
            return fused_gdr_h(
                k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                initial_state=None, output_final_state=True, output_h=False,
                cu_seqlens=cp_cu, num_warmup_chunks=nw_bidi,
            )

        def run_recompute_h():
            return fused_gdr_h(
                k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                initial_state=cp_h0, output_final_state=False, output_h=True,
                cu_seqlens=cp_cu,
            )

        def run_correct_h0():
            return correct_initial_states(
                raw_h0=None, ht_buffer=ht, mt_buffer=mt,
                fallback_mask=fb_mask, seq_map_r2c=ctx.seq_map_r2c,
            )

        def run_fused_fwd():
            return fused_gdr_fwd(
                q=inp["q"], k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                scale=inp["scale"], initial_state=cp_h0,
                output_final_state=True, output_h=False, output_o=True,
                cu_seqlens=cp_cu, cp_seq_map=ctx.seq_map_c2r,
                raw_cu_seqlens=cu_raw,
            )

        def run_prepare_dh():
            return fused_gdr_dh(
                q=inp["q"], k=inp["k"], a=A, g=g_c, b=inp["beta"], do=inp["do"],
                dht=None, output_dh0=True, output_dh=False, scale=inp["scale"],
                cu_seqlens=cp_cu, num_warmup_chunks=nw_bwd,
            )

        def run_correct_dht():
            return correct_terminal_states(
                raw_dht=None, dht_buffer=dht_buffer, mt_buffer=mt,
                fallback_mask=fb_bwd, seq_map_r2c=ctx.seq_map_r2c,
            )

        def run_fused_bwd():
            return fused_gdr_bwd(
                q=inp["q"], k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                do=inp["do"], dht=None, h=h_states, scale=inp["scale"],
                cu_seqlens=cp_cu,
            )

        closures = {
            "prepare_h": run_prepare_h, "prepare_h_bidi": run_prepare_h_bidi,
            "correct_h0": run_correct_h0, "correct_dht": run_correct_dht,
            "fused_fwd": run_fused_fwd, "prepare_dh": run_prepare_dh,
            "recompute_h": run_recompute_h, "fused_bwd": run_fused_bwd,
        }
        assert set(closures) == set(MEASURED), (
            f"timed closures must cover MEASURED exactly; "
            f"missing {sorted(set(MEASURED) - set(closures))}, "
            f"extra {sorted(set(closures) - set(MEASURED))}")
        t = {name: bench(closures[name], warmup_ms, rep_ms) for name in MEASURED}

        yield dict(
            name=spec.name, device=device, P=int(P), chunk=int(chunk),
            dtype=spec.dtype, swa_ratio=f"{spec.swa_ratio:g}",
            H=spec.H, Hk=spec.Hk, T=spec.T, Lc=sum(chunks), L_cp=int(lcp),
            N_part=n_part, B_raw=spec.B_raw, cmax=max(chunks), cu=spec.cu_str,
            warmup_max=int(nw.max().item()),
            warmup_mean=float(nw.float().mean().item()),
            warmup_bidi_max=int(nw_bidi.max().item()),
            warmup_bidi_mean=float(nw_bidi.float().mean().item()),
            warmup_bwd_max=int(nw_bwd.max().item()),
            warmup_bwd_mean=float(nw_bwd.float().mean().item()),
            gate_avg=gate_avg_str,
            **t,
            # Totals are over launches, not coefficient rows: correct runs twice.
            fwd_total=t["prepare_h"] + t["correct_h0"] + t["fused_fwd"],
            all_total=(t["prepare_h_bidi"] + t["correct_h0"] + t["fused_fwd"]
                       + t["prepare_dh"] + t["correct_dht"]
                       + t["recompute_h"] + t["fused_bwd"]),
        )


def plan_cases(specs, *, P: int, chunk: int, ratio: float, device: str,
               cache: dict, refresh: bool = False) -> tuple[list[tuple], int]:
    todo, hits = [], 0
    for spec in specs:
        want = []
        for lcp in lcp_grid(spec.cmax(chunk), ratio):
            key = case_key(spec, lcp, P=P, chunk=chunk, device=device)
            row = None if refresh else cache.get(key)
            if row is not None and cache_row_complete(row):
                hits += 1
            else:
                # present but pre-dating a column is not usable: re-measure it.
                want.append(lcp)
        if want:
            todo.append((spec, want))
    return todo, hits


def measure_specs(specs, *, cache_path: str, P: int, chunk: int = CHUNK_SIZE,
                  ratio: float | None = None, device: str | None = None,
                  warmup_ms: float = 25, rep_ms: float = 100, seed: int = 42,
                  refresh: bool = False, verbose: bool = True) -> dict[tuple, dict]:
    from .utils import DEFAULT_LCP_RATIO
    if ratio is None:
        ratio = DEFAULT_LCP_RATIO
    if device is None:
        device = torch.cuda.get_device_name()
    cache = read_cache(cache_path)
    todo, hits = plan_cases(specs, P=P, chunk=chunk, ratio=ratio,
                            device=device, cache=cache, refresh=refresh)
    n_todo = sum(len(l) for _, l in todo)
    n_dropped = 0
    if refresh and todo:
        redo = {case_key(spec, lcp, P=P, chunk=chunk, device=device)
                for spec, lcps in todo for lcp in lcps}
        n_dropped = len(redo & cache.keys())
        cache = {k: r for k, r in cache.items() if k not in redo}
        rewrite_cache(cache_path, cache.values())
    if verbose:
        note = f", {n_dropped} existing row(s) dropped" if n_dropped else ""
        print(f"cache: {cache_path} ({hits} hit, {n_todo} to measure{note})")
    done = 0
    for spec, lcps in todo:
        for row in measure_shape(spec, lcps, P=P, chunk=chunk, device=device,
                                 warmup_ms=warmup_ms, rep_ms=rep_ms, seed=seed):
            append_cache(cache_path, row)
            cache[cache_key(row)] = row
            done += 1
            if verbose:
                print(f"  [{done}/{n_todo}] {spec.name:12s} L_cp={row['L_cp']:<5d} "
                      f"N_part={row['N_part']:<5d} all_total={row['all_total']:.4f} ms")
    return cache
