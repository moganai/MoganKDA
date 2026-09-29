# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.launch import coef_rows
from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    COEFS_DIR,
    COEF_COLUMNS,
    CP_ONLY_KERNELS,
    KERNELS,
    KERNEL_SOURCES,
    MIN_LCP,
    seq_chunks,
)

# Timed columns, one per call site. KERNEL_SOURCES maps kernels back to these.
MEASURED = ("prepare_h", "prepare_h_bidi", "correct_h0", "correct_dht",
            "fused_fwd", "prepare_dh", "recompute_h", "fused_bwd")


def save_coefs(path: str, coefs: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", *COEF_COLUMNS, "P", "chunk", "arch"])
        for row in coef_rows(coefs["backend"], KERNELS):
            k = coefs["kernels"][row]
            w.writerow([row, *(f"{k[c]:.8g}" for c in COEF_COLUMNS),
                        coefs["P"], coefs["chunk"], coefs["arch"]])


# ---------------------------------------------------------------------------
# The L_cp grid
# ---------------------------------------------------------------------------
DEFAULT_LCP_RATIO = 1.5


def lcp_grid(cmax: int, ratio: float = DEFAULT_LCP_RATIO,
             min_S: int = MIN_LCP) -> list[int]:
    cmax = int(cmax)
    if cmax <= min_S:
        return [cmax]
    grid, s = [], int(min_S)
    while s < cmax:
        grid.append(s)
        # round() alone stalls at small S, where ratio is under one chunk.
        s = max(s + 1, int(round(s * ratio)))
    grid.append(cmax)
    return grid


# ---------------------------------------------------------------------------
# Shape CSV: what the fit takes as input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeSpec:
    name: str
    H: int                    # value heads
    Hk: int                   # key heads
    cu: tuple[int, ...]       # cumulative token offsets, cu[0] == 0
    dtype: str = "bfloat16"
    swa_ratio: float = 0.0

    @property
    def seqlens(self) -> tuple[int, ...]:
        return tuple(self.cu[i + 1] - self.cu[i] for i in range(len(self.cu) - 1))

    @property
    def T(self) -> int:
        return int(self.cu[-1])

    @property
    def B_raw(self) -> int:
        return len(self.cu) - 1

    def chunks(self, chunk: int) -> list[int]:
        return seq_chunks(self.cu, chunk)

    def cmax(self, chunk: int) -> int:
        return max(self.chunks(chunk))

    @property
    def cu_str(self) -> str:
        return ":".join(str(int(x)) for x in self.cu)


def _parse_cu(row: dict, name: str) -> tuple[int, ...]:
    if row.get("cu"):
        cu = [int(x) for x in str(row["cu"]).replace(",", ":").split(":")]
        if cu[0] != 0:
            raise ValueError(f"{name}: 'cu' must start at 0, got {cu[0]}")
        lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    elif row.get("seqlens"):
        lens = [int(x) for x in str(row["seqlens"]).replace(":", ",").split(",")]
    elif row.get("seqlen"):
        lens = [int(row["seqlen"])] * int(row.get("batch") or 1)
    else:
        raise ValueError(
            f"{name}: need one of 'cu' / 'seqlens' / 'seqlen' (+ optional 'batch')")
    if not lens or min(lens) <= 0:
        raise ValueError(f"{name}: non-positive sequence length in {lens}")
    out, acc = [0], 0
    for x in lens:
        acc += x
        out.append(acc)
    return tuple(out)


def load_specs(path: str, *, Hk: int | None = None, dtype: str = "bfloat16",
               swa_ratio: float = 0.0) -> list[ShapeSpec]:
    specs: list[ShapeSpec] = []
    with open(path, newline="") as f:
        rows = (ln for ln in f if not ln.lstrip().startswith("#"))
        for i, row in enumerate(csv.DictReader(rows)):
            row = {(k.strip() if k else k): v for k, v in row.items()}
            if not any((v or "").strip() for v in row.values()):
                continue  # blank line
            if not row.get("H"):
                raise ValueError(
                    f"{path}: data row {i + 1} "
                    f"({row.get('name') or 'unnamed'}) has no 'H'")
            H = int(row["H"])
            fallback = f"row{i + 1}"
            cu = _parse_cu(row, row.get("name") or fallback)
            B = len(cu) - 1
            specs.append(ShapeSpec(
                name=(row.get("name") or "").strip() or f"h{H}_{B}x{cu[-1] // B}",
                H=H,
                Hk=int(row["Hk"]) if row.get("Hk") else (Hk if Hk else H),
                cu=cu,
                dtype=(row.get("dtype") or "").strip() or dtype,
                swa_ratio=(float(row["swa_ratio"])
                           if row.get("swa_ratio") not in (None, "") else swa_ratio),
            ))
    if not specs:
        raise ValueError(f"no shape rows in {path}")
    dupes = sorted({s.name for s in specs
                    if sum(x.name == s.name for x in specs) > 1})
    if dupes:
        raise ValueError(
            f"{path}: duplicate shape names {dupes}. Names are the report's "
            f"grouping key (and the unit of regret / verdict), so they must be "
            f"unique -- add or edit the 'name' column.")
    return specs


def parse_gate_avg(s) -> list[float]:
    """Parse a cache row's ``gate_avg`` field (``measure.py`` writes it as a
    ``:``-joined string, one decay average per value head)."""
    return [float(x) for x in str(s).split(":") if x != ""]


# ---------------------------------------------------------------------------
# Measurement cache: what sits between measuring and fitting
# ---------------------------------------------------------------------------
CACHE_COLUMNS = (
    "name", "device", "P", "chunk", "dtype", "swa_ratio",
    "H", "Hk", "T", "Lc", "L_cp", "N_part", "B_raw", "cmax", "cu",
    "warmup_max", "warmup_mean",
    "warmup_bidi_max", "warmup_bidi_mean", "warmup_bwd_max", "warmup_bwd_mean",
    "gate_avg",
    *MEASURED,
    "fwd_total", "all_total",
)

# name is deliberately absent: relabelling a shape must not invalidate its timings.
CACHE_KEY_COLUMNS = ("device", "P", "chunk", "dtype", "swa_ratio",
                     "H", "Hk", "cu", "L_cp")

_KEY_FMT = {"P": int, "chunk": int, "H": int, "Hk": int, "L_cp": int,
            "swa_ratio": lambda v: f"{float(v):g}"}


def cache_key(row: dict) -> tuple:
    out = []
    for col in CACHE_KEY_COLUMNS:
        v = row[col]
        f = _KEY_FMT.get(col)
        out.append(str(f(v)) if f else str(v).strip())
    return tuple(out)


def case_key(spec, lcp: int, *, P: int, chunk: int, device: str) -> tuple:
    """The cache key of one (shape, L_cp) case. Both the measurement planner and
    the fitter ask for rows this way, so they have to agree on it."""
    return cache_key(dict(device=device, P=P, chunk=chunk, dtype=spec.dtype,
                          swa_ratio=spec.swa_ratio, H=spec.H, Hk=spec.Hk,
                          cu=spec.cu_str, L_cp=lcp))


def read_cache(path: str) -> dict[tuple, dict]:
    if not path or not os.path.exists(path):
        return {}
    out: dict[tuple, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            missing = [c for c in CACHE_KEY_COLUMNS if c not in row]
            if missing:
                raise ValueError(
                    f"{path} is not an autocp measurement cache "
                    f"(missing key columns {missing})")
            for col in ("warmup_mean", "warmup_bidi_mean", "warmup_bwd_mean",
                        *MEASURED, "fwd_total", "all_total"):
                if row.get(col) not in (None, ""):
                    row[col] = float(row[col])
            for col in ("H", "Hk", "T", "Lc", "L_cp", "N_part", "B_raw", "cmax",
                        "warmup_max", "warmup_bidi_max", "warmup_bwd_max",
                        "P", "chunk"):
                if row.get(col) not in (None, ""):
                    row[col] = int(float(row[col]))
            out[cache_key(row)] = row
    return out


def cache_row_complete(row: dict) -> bool:
    return all(isinstance(row.get(c), float) for c in MEASURED)


def rewrite_cache(path: str, rows) -> None:
    rows = list(rows)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CACHE_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def _cache_header(path: str) -> list[str] | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as f:
        for header in csv.reader(f):
            return [h.strip() for h in header]
    return None


def append_cache(path: str, row: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    header = _cache_header(path)
    if header is not None and header != list(CACHE_COLUMNS):
        extra = [c for c in header if c not in CACHE_COLUMNS]
        missing = [c for c in CACHE_COLUMNS if c not in header]
        raise ValueError(
            f"{path} was written under a different cache schema, so appending "
            f"would silently write values under the wrong column names"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + (" (same columns, different order)"
               if not missing and not extra else "")
            + ". Measure into a new cache file instead.")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CACHE_COLUMNS),
                          extrasaction="ignore")
        if header is None:
            w.writeheader()
        w.writerow(row)
        f.flush()


def device_slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()
                   ).strip("_") or "unknown"

DEBUG_DIR = "tmp"


def default_cache_path(device: str, root: str = DEBUG_DIR) -> str:
    return os.path.join(root, f"autocp_cache_{device_slug(device)}.csv")


def default_coefs_path(arch: str, root: str = DEBUG_DIR) -> str:
    return os.path.join(root, f"coefs_{arch}.csv")


def shipped_coefs_path(arch: str) -> str:
    return os.path.join(COEFS_DIR, f"{arch}.csv")

def depth_col(col: str) -> str:
    """Column holding the schedule depth of measured column ``col``."""
    return f"d_{col}"


def row_col(col: str) -> str:
    """Column holding the coefficient row ``col``'s launch routes to."""
    return f"row_{col}"

MODES = {"infer": "fwd_total", "train": "all_total"}

SOURCE_OF = {col: kernel for kernel, modes in KERNEL_SOURCES.items()
             for col in modes.values()}
MODE_OF = {col: mode == "train" for modes in KERNEL_SOURCES.values()
           for mode, col in modes.items()}
STEP_SITES = {mode: tuple((kernel, modes[mode])
                          for kernel, modes in KERNEL_SOURCES.items()
                          if mode in modes)
              for mode in MODES}

FEATURE_COLUMNS = ("u", *(depth_col(col) for col in MEASURED))


def check_taxonomy() -> None:
    if set(SOURCE_OF) != set(MEASURED):
        raise AssertionError(
            f"every measured column must belong to exactly one kernel; "
            f"unclaimed {sorted(set(MEASURED) - set(SOURCE_OF))}")
    stray = {m for modes in KERNEL_SOURCES.values() for m in modes} - set(MODES)
    if stray:
        raise AssertionError(f"KERNEL_SOURCES names modes {sorted(stray)} that "
                             f"MODES does not: {sorted(MODES)}")


check_taxonomy()


def priced_rows(df: pd.DataFrame, kernel: str) -> pd.DataFrame:
    return df[df["cp"]] if kernel in CP_ONLY_KERNELS else df


def design(df: pd.DataFrame, col: str):
    X = np.stack([df[depth_col(col)].to_numpy(float), df["u"].to_numpy(float),
                  np.ones(len(df))], axis=1)
    return X, df[col].to_numpy(float)


def predict_column(coefs: dict, df: pd.DataFrame, col: str) -> np.ndarray:
    X, _ = design(df, col)
    C = np.array([[coefs["kernels"][r][c] for c in COEF_COLUMNS]
                  for r in df[row_col(col)]])
    return (X * C).sum(axis=1)


def mape(y: np.ndarray, yhat: np.ndarray, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])) * 100)
