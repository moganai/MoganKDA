# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.launch import (
    BACKEND_OF,
    coef_rows,
    current_arch,
    launch_config,
)
from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    COEF_COLUMNS,
    KERNELS,
    seq_chunks,
    struct_features,
    warmup_from_gate_avg,
)
from .utils import (
    CP_ONLY_KERNELS,
    DEFAULT_LCP_RATIO,
    FEATURE_COLUMNS,
    MEASURED,
    MODE_OF,
    MODES,
    SOURCE_OF,
    cache_row_complete,
    case_key,
    default_cache_path,
    default_coefs_path,
    depth_col,
    design,
    lcp_grid,
    load_specs,
    parse_gate_avg,
    priced_rows,
    row_col,
    save_coefs,
    shipped_coefs_path,
)
from .measure import measure_specs
from .plots import ensure_matplotlib, plot_fit
from .reports import accuracy_report, decision_report

__all__ = ["build_frame", "rows_for_specs", "fit_row", "fit_all", "main"]


# ---------------------------------------------------------------------------
# Cache rows -> feature frame
# ---------------------------------------------------------------------------
def rows_for_specs(cache: dict, specs, *, P: int, chunk: int, ratio: float,
                   device: str, strict: bool = True) -> list[dict]:
    rows, missing = [], []
    for spec in specs:
        for lcp in lcp_grid(spec.cmax(chunk), ratio):
            key = case_key(spec, lcp, P=P, chunk=chunk, device=device)
            row = cache.get(key)
            if row is None or not cache_row_complete(row):
                missing.append((spec.name, lcp))
            else:
                rows.append({**row, "name": spec.name})
    if missing and strict:
        raise SystemExit(
            f"{len(missing)} measurement(s) missing or incomplete in the cache "
            f"(a row is incomplete when any of {len(MEASURED)} timed columns is "
            f"absent), e.g. {missing[:6]}. Re-run with --refresh-cache to "
            f"re-measure them.")
    return rows


def build_frame(rows, P: int, chunk: int = 64, part: str | None = None,
                *, backend: str) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("no measured rows")
    df["setting"] = df["name"]
    if part is not None:
        df["part"] = part

    # Features twice per row, once per mode: only the warmup depths differ
    # -- fwd-only for inference, bidi max for training.
    args = [(seq_chunks([int(x) for x in str(r["cu"]).split(":")], chunk),
             int(r["L_cp"]), int(r["H"]),
             warmup_from_gate_avg(parse_gate_avg(r["gate_avg"]), chunk))
            for _, r in df.iterrows()]
    feats = [struct_features(c, S, H, P, warmup_per_head=w, is_train=False,
                             backend=backend) for c, S, H, w in args]
    feats_tr = [struct_features(c, S, H, P, warmup_per_head=w, is_train=True,
                                backend=backend) for c, S, H, w in args]

    df["u"] = [f.u for f in feats]
    df["cp"] = [f.enable_cp for f in feats]
    for col, is_train in MODE_OF.items():
        kernel = SOURCE_OF[col]
        fs = feats_tr if is_train else feats
        df[depth_col(col)] = [f.depth[kernel] for f in fs]
        df[row_col(col)] = [launch_config(backend, kernel, f.shape).row for f in fs]
    missing = [c for c in FEATURE_COLUMNS if c not in df]
    if missing:
        raise AssertionError(f"build_frame did not produce {missing}")

    if "N_part" in df:
        got = df["N_part"].astype(int).to_numpy()
        want = np.array([f.num_partitions for f in feats])
        bad = np.flatnonzero(got != want)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"{bad.size} row(s) where the measured partition count differs "
                f"from struct_features, e.g. {df['setting'].iloc[i]} "
                f"L_cp={df['L_cp'].iloc[i]}: measured {got[i]} vs modelled "
                f"{want[i]}. depth_runs and _build_intra_cp_context disagree.")
    return df


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------
def _row_blocks(df: pd.DataFrame, row: str):
    """(X, y) per measured column whose launch routes that column to `row`."""
    blocks = []
    for col, kernel in SOURCE_OF.items():
        sub = priced_rows(df, kernel)
        sub = sub[sub[row_col(col)] == row]
        if len(sub):
            blocks.append(design(sub, col))
    return blocks


def row_counts(df: pd.DataFrame, backend: str) -> dict[str, int]:
    return {row: sum(len(y) for _, y in _row_blocks(df, row))
            for row in coef_rows(backend, KERNELS)}


def fit_row(df: pd.DataFrame, row: str, settings=None) -> np.ndarray:
    sub = df if settings is None else df[df["setting"].isin(settings)]
    blocks = _row_blocks(sub, row)
    n = sum(len(y) for _, y in blocks)
    if n < len(COEF_COLUMNS) + 1:
        raise SystemExit(
            f"row {row!r} has only {n} measurement(s) to fit "
            f"{len(COEF_COLUMNS)} coefficients. Widen the shape set (or the "
            f"L_cp grid) so both sides of its launch rule are covered.")
    X = np.concatenate([b[0] for b in blocks], axis=0)
    y = np.concatenate([b[1] for b in blocks], axis=0)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef[0] = max(coef[0], 0.0)
    return coef


def fit_all(df: pd.DataFrame, P: int, chunk: int = 64, *, arch: str) -> dict:
    backend = BACKEND_OF[arch]
    kernels = {row: dict(zip(COEF_COLUMNS,
                             (float(x) for x in fit_row(df, row))))
               for row in coef_rows(backend, KERNELS)}
    return {"kernels": kernels, "P": int(P), "chunk": int(chunk),
            "arch": arch, "backend": backend}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Artifacts:
    coefs: dict
    counts: dict
    acc: pd.DataFrame
    decs: dict
    allrows: pd.DataFrame


def _resolve_env(args) -> tuple[str, int]:
    """device / P for the cache key and the model: --device / --sm-count when
    given, else this GPU's -- which is also what mogan_kda was compiled for."""
    device, P = args.device, args.sm_count
    if device is None or P is None:
        import torch
        if device is None:
            device = torch.cuda.get_device_name()
        if P is None:
            P = torch.cuda.get_device_properties().multi_processor_count
    return device, int(P)


def _resolve_arch(args) -> str:
    """--arch when given, else this GPU's via the same probe mogan_kda runs on
    import (so it is already cached here).

    Only the omitted case asks the device, which keeps --arch a pure override.
    A box with no visible GPU dies earlier than this -- importing mogan_kda
    already queries it -- so the guard mainly keeps the message legible.
    """
    if args.arch:
        return args.arch
    try:
        arch = current_arch()
    except Exception as exc:  # no GPU, or a compute version we don't support
        raise SystemExit(
            f"could not detect the architecture: {exc}\n"
            f"pass --arch explicitly ({'|'.join(sorted(BACKEND_OF))}), or "
            f"ARCH=<arch> when running through scripts/fit_autocp.sh") from exc
    print(f"arch: detected {arch} on this device")
    return arch


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="fit the autocp latency model from shape CSVs "
                    "(measuring the kernels as needed)")
    ap.add_argument("--train", required=True, help="training shape CSV")
    ap.add_argument("--arch", default=None, choices=sorted(BACKEND_OF),
                    help="architecture these coefficients are for; picks the "
                         "backend whose block_DV rules route the fitted rows "
                         "(default: this GPU's)")
    ap.add_argument("--eval", dest="eval_path", default=None,
                    help="held-out shape CSV (accuracy + decision quality only)")
    ap.add_argument("--out", default=None,
                    help="output coefficient CSV (default "
                         "tmp/coefs_<arch>.csv, never the in-package one)")
    ap.add_argument("--cache", default=None,
                    help="measurement cache CSV "
                         "(default tmp/autocp_cache_<device>.csv)")
    ap.add_argument("--sm-count", type=int, default=None,
                    help="P; defaults to this device's SM count")
    ap.add_argument("--device", default=None,
                    help="device name for the cache key; defaults to this GPU")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lcp-ratio", type=float, default=None,
                    help="geometric step of the L_cp grid (default 1.5)")
    ap.add_argument("--Hk", type=int, default=None,
                    help="key heads when the CSV has no Hk column (default: H)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--swa-ratio", type=float, default=0.0)
    ap.add_argument("--warmup-ms", type=float, default=25)
    ap.add_argument("--rep-ms", type=float, default=100)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="re-measure every case instead of reusing cache hits "
                         "(the re-timed rows replace their stale ones)")
    ap.add_argument("--report-csv", default=None,
                    help="also write the accuracy / decision tables here "
                         "(<stem>_accuracy.csv, <stem>_decision.csv)")
    ap.add_argument("--plot", default=None, metavar="DIR",
                    help="write curves_<mode>.png / parity.png here (needs "
                         "matplotlib, which is not a mogan_kda dependency)")
    args = ap.parse_args(argv)

    if args.lcp_ratio is None:
        args.lcp_ratio = DEFAULT_LCP_RATIO
    args.arch = _resolve_arch(args)
    if not args.out:
        args.out = default_coefs_path(args.arch)
    if args.plot:
        ensure_matplotlib()  # fail on a missing matplotlib now, not after measuring
    return args


def run(args) -> Artifacts:
    kw = dict(Hk=args.Hk, dtype=args.dtype, swa_ratio=args.swa_ratio)
    groups = [("train", load_specs(args.train, **kw))]
    if args.eval_path:
        groups.append(("eval", load_specs(args.eval_path, **kw)))
    clash = ({s.name for s in groups[0][1]} & {s.name for s in groups[-1][1]}
             if len(groups) > 1 else set())
    if clash:
        raise SystemExit(f"shape names appear in both --train and --eval: "
                         f"{sorted(clash)}")

    device, P = _resolve_env(args)
    cache_path = args.cache or default_cache_path(device)

    backend = BACKEND_OF[args.arch]
    specs = [s for _, g in groups for s in g]
    cache = measure_specs(specs, cache_path=cache_path, P=P,
                          chunk=args.chunk, ratio=args.lcp_ratio,
                          device=device, warmup_ms=args.warmup_ms,
                          rep_ms=args.rep_ms, refresh=args.refresh_cache)

    frames = []
    for part, g in groups:
        rows = rows_for_specs(cache, g, P=P, chunk=args.chunk,
                              ratio=args.lcp_ratio, device=device)
        df = build_frame(rows, P, args.chunk, part=part, backend=backend)
        frames.append(df)
        print(f"{part + ':':6s} {len(g)} shapes, {len(df)} rows "
              f"({int(df['cp'].sum())} CP, "
              f"{int((~df['cp']).sum())} baseline)")
    train = frames[0]
    allrows = pd.concat(frames, ignore_index=True)

    coefs = fit_all(train, P, args.chunk, arch=args.arch)
    counts = row_counts(train, backend)
    print(f"\n=== coefficients ({args.arch}/{backend}, P={P}, "
          f"chunk={args.chunk}) ===")
    print(f"fitted on train only; baseline rows held out of "
          f"{', '.join(CP_ONLY_KERNELS)}")
    for row, c in coefs["kernels"].items():
        print(f"  {row:17s} a={c['a']:.7f}  b={c['b']:.7f}  "
              f"c={c['c']:.6f}   n={counts[row]}")

    acc = accuracy_report(allrows, coefs)
    print("\n=== MAPE% per setting x call site (rows that price it) ===")
    print(acc.round(2).to_string(index=False))
    print("\nmean by part x call site:")
    print(acc.groupby(["part", "site"])["mape"].mean().unstack()
          .round(2).to_string())

    decs = {}
    for mode in MODES:
        dec = decision_report(allrows, coefs, P, mode)
        decs[mode] = dec
        print(f"\n=== decision quality, {mode} "
              f"(model vs the measured {MODES[mode]} grid) ===")
        print(dec.round(2).to_string(index=False))
        bad = dec[dec["verdict"] != dec["verdict_meas"]].dropna(
            subset=["verdict_meas"])
        if len(bad):
            print(f"!! {len(bad)} setting(s) where the {mode} verdict disagrees "
                  f"with the measurement: {sorted(bad['setting'])}")

    return Artifacts(coefs=coefs, counts=counts, acc=acc, decs=decs,
                     allrows=allrows)


def _mkdirs_for(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _print_ship_hint(path: str, arch: str) -> None:
    """Reminder that a fit is not installed. Kernel times drift between sessions,
    so overwriting the shipped CSV stays a deliberate `cp`."""
    shipped = shipped_coefs_path(arch)
    if os.path.realpath(path) == os.path.realpath(shipped):
        return
    # absolute, so the two lines survive being pasted into a shell elsewhere
    path = os.path.abspath(path)
    print("\nnot installed. compare, then ship:")
    print(f"  diff {path} {shipped}")
    print(f"  cp   {path} {shipped}")


def emit(args, art: Artifacts) -> None:
    if args.out:
        _mkdirs_for(args.out)
        save_coefs(args.out, art.coefs)
        print(f"\ncoefficients written: {args.out}")
    if args.report_csv:
        _mkdirs_for(args.report_csv)
        stem = os.path.splitext(args.report_csv)[0]
        art.acc.to_csv(f"{stem}_accuracy.csv", index=False)
        # both modes in one file, distinguished by the mode column
        pd.concat(art.decs.values(), ignore_index=True).to_csv(
            f"{stem}_decision.csv", index=False)
        print(f"reports written: {stem}_accuracy.csv, {stem}_decision.csv")
    # last, so a plotting failure can't cost a coefficient run its GPU time
    if args.plot:
        for p in plot_fit(art.allrows, art.coefs, art.decs, args.plot):
            print(f"figure written: {p}")
    if args.out:  # the ship reminder reads best as the final line
        _print_ship_hint(args.out, args.arch)


def main(argv=None) -> None:
    args = parse_args(argv)
    emit(args, run(args))


if __name__ == "__main__":
    main()
