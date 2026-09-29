# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .utils import (
    CP_ONLY_KERNELS,
    MEASURED,
    SOURCE_OF,
    STEP_SITES,
    mape,
    predict_column,
    priced_rows,
    row_col,
)

__all__ = ["ensure_matplotlib", "plot_fit"]

# Panel tint per part (held-out eval shapes read differently from train).
_PART_BG = {"train": "#ffffff", "eval": "#eef4ee"}

# One colour per call site; two sources of one row get two tints of one hue.
_SITE_COLOR = {"prepare_h": "#4C78A8", "prepare_h_bidi": "#8AB4D8",
               "correct_h0": "#F58518", "correct_dht": "#FBBE79",
               "fused_fwd": "#54A24B", "fused_bwd": "#E45756",
               "prepare_dh": "#B279A2", "recompute_h": "#9D755D"}


def ensure_matplotlib():
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit(
            "--plot needs matplotlib, which is not a dependency of mogan_kda "
            "(nothing on the launch path plots). Install it for calibration "
            "only:\n    pip install matplotlib") from e
    matplotlib.use("Agg")  # calibration runs on headless GPU boxes
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _parts(df: pd.DataFrame) -> pd.Series:
    return df["part"] if "part" in df else pd.Series("train", index=df.index)


def _totals(coefs: dict, df: pd.DataFrame, mode: str = "train"):
    sites = STEP_SITES[mode]
    base_sites = tuple(s for s in sites if s[0] not in CP_ONLY_KERNELS)
    meas = lambda ss: df[[c for _, c in ss]].sum(axis=1).to_numpy(float)
    pred = lambda ss: sum(predict_column(coefs, df, c) for _, c in ss)
    cp = df["cp"].to_numpy(bool)
    return (np.where(cp, meas(sites), meas(base_sites)),
            np.where(cp, pred(sites), pred(base_sites)))


def _plot_curves(plt, matplotlib, df: pd.DataFrame, dec: pd.DataFrame,
                 path: str, mode: str = "train") -> str:
    rec = {r["setting"]: r for r in dec.to_dict("records")}
    shapes = sorted(df["setting"].unique())
    ncol = min(4, len(shapes))
    nrow = -(-len(shapes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), shapes):
        g = df[df["setting"] == name].sort_values("L_cp")
        cp, base = g[g["cp"]], g[~g["cp"]]
        part = str(_parts(g).iloc[0])
        r = rec.get(name, {})
        if len(cp):
            ax.plot(cp["L_cp"], cp["meas_total"], "o", ms=4.5, color="#333",
                    label="measured (CP)")
            ax.plot(cp["L_cp"], cp["pred_total"], "-", lw=1.7, color="#E45756",
                    label="model")
        if len(base):
            ax.axhline(float(base["meas_total"].iloc[0]), color="#4C78A8",
                       lw=1.3, label="measured no-CP")
            ax.axhline(float(base["pred_total"].iloc[0]), color="#4C78A8",
                       lw=1.1, ls="--", label="model no-CP")
        for key, color in (("lcp_meas", "#333"), ("lcp_model", "#E45756")):
            if r.get(key):
                ax.axvline(int(r[key]), color=color, ls=":", lw=1.1)
        regret = r.get("regret_pct")
        sub = (f"L* meas {r.get('lcp_meas', '-')} / model "
               f"{r.get('lcp_model', '-')}"
               + (f", regret {regret:.2f}%" if regret == regret else "")
               + f"\nverdict {r.get('verdict', '?')}"
               + (f" (measured {r['verdict_meas']})" if r.get("verdict_meas")
                  else ""))
        ax.set(xscale="log", xlabel="L_cp (chunks)", ylabel="latency (ms)",
               title=f"{name}  [{part}]\n{sub}")
        ax.title.set_fontsize(9)
        # tick the shape's own grid as plain chunk counts, thinned to <=6 labels
        grid = sorted(int(x) for x in g["L_cp"].unique())
        ax.set_xticks(grid[::-(-len(grid) // 6)])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_facecolor(_PART_BG.get(part, "#ffffff"))
        ax.grid(alpha=.25, which="both")
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7)
    for ax in axes.ravel()[len(shapes):]:
        ax.axis("off")
    fig.suptitle(f"autocp [{mode}]: fitted latency vs L_cp per shape "
                 f"(green panels = held-out eval shapes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_parity(plt, matplotlib, df: pd.DataFrame, coefs: dict,
                 path: str) -> str:
    ncol = min(3, len(MEASURED))
    nrow = -(-len(MEASURED) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.75 * ncol, 4.5 * nrow),
                            squeeze=False)
    parts = _parts(df).to_numpy()
    for ax, k in zip(axes.ravel(), MEASURED):
        kernel = SOURCE_OF[k]
        priced = df.index.isin(priced_rows(df, kernel).index)
        y = df[k].to_numpy(float)
        yhat = predict_column(coefs, df, k)
        tags = []
        # eval drawn hollow and on top so it isn't buried under the train points
        for part, style in (
                ("train", dict(marker="o", s=24, alpha=.7,
                               c=_SITE_COLOR[k])),
                ("eval", dict(marker="*", s=75, facecolors="none",
                              edgecolors="k", linewidths=.8, zorder=5))):
            m = priced & (parts == part)
            if not m.any():
                continue
            ax.scatter(y[m], yhat[m], label=f"{part} (n={int(m.sum())})", **style)
            tags.append(f"{part} {mape(y, yhat, m):.1f}%")
        lo, hi = y[priced].min() * .7, y[priced].max() * 1.3
        ax.plot([lo, hi], [lo, hi], "k-", lw=.8)
        for f in (1.1, 1 / 1.1):
            ax.plot([lo, hi], [lo * f, hi * f], "k--", lw=.6, alpha=.45)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel="measured (ms)", ylabel="predicted (ms)",
               title=f"{k}   [{', '.join(sorted(df.loc[priced, row_col(k)].unique()))}]"
                     f"\nMAPE  {'   '.join(tags)}")
        ax.title.set_fontsize(10)
        for axis in (ax.xaxis, ax.yaxis):  # log minor labels collide otherwise
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.25, which="both")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(MEASURED):]:
        ax.axis("off")
    fig.suptitle("autocp latency model: per-call-site parity, fitted on train only "
                 "(dashed = +/-10%; sources sharing a row share a hue)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fit(df: pd.DataFrame, coefs: dict, decs: dict,
             outdir: str) -> list[str]:
    matplotlib, plt = ensure_matplotlib()
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for mode, dec in decs.items():
        d = df.copy()
        d["meas_total"], d["pred_total"] = _totals(coefs, d, mode)
        paths.append(_plot_curves(plt, matplotlib, d, dec,
                                  os.path.join(outdir, f"curves_{mode}.png"),
                                  mode))
    paths.append(_plot_parity(plt, matplotlib, df, coefs,
                              os.path.join(outdir, "parity.png")))
    return paths
