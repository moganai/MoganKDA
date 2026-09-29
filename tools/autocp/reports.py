# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import pandas as pd

from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.decision import decide
from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    warmup_from_gate_avg,
)

from .utils import (
    CP_ONLY_KERNELS,
    MEASURED,
    MODES,
    SOURCE_OF,
    STEP_SITES,
    mape,
    parse_gate_avg,
    predict_column,
    priced_rows,
)

__all__ = ["accuracy_report", "nocp_measured", "decision_report"]


def accuracy_report(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    keys = (["part", "setting"] if "part" in df else ["setting"])
    out = []
    for col in MEASURED:
        kernel = SOURCE_OF[col]
        y = df[col].to_numpy(float)
        yhat = predict_column(coefs, df, col)
        priced = df.index.isin(priced_rows(df, kernel).index)
        for vals, g in df.groupby(keys, sort=False):
            vals = vals if isinstance(vals, tuple) else (vals,)
            sel = df.index.isin(g.index) & priced
            out.append({**dict(zip(keys, vals)), "site": col, "kernel": kernel,
                        "mape": mape(y, yhat, sel), "n_fit": int(sel.sum())})
    return pd.DataFrame(out).sort_values([*keys, "site"], kind="stable")


def nocp_measured(base_rows: pd.DataFrame, mode: str = "train") -> float:
    cols = [c for k, c in STEP_SITES[mode] if k not in CP_ONLY_KERNELS]
    return float(sum(base_rows[c].iloc[0] for c in cols))


def decision_report(df: pd.DataFrame, coefs: dict, P: int,
                    mode: str = "train") -> pd.DataFrame:
    total_col = MODES[mode]
    out = []
    for label, sub in df.groupby("setting", sort=True):
        H = int(sub["H"].iloc[0])
        chunk = int(sub["chunk"].iloc[0])
        cu = [int(x) for x in str(sub["cu"].iloc[0]).split(":")]
        cp = sub[sub["cp"]]
        base = sub[~sub["cp"]]
        kw = dict(cu_seqlens=cu, num_v_heads=H, coefs=coefs, P=P, debug=True,
                  is_train=(mode == "train"),
                  warmup_per_head=warmup_from_gate_avg(
                      parse_gate_avg(sub["gate_avg"].iloc[0]), chunk))

        rec = dict(setting=label, mode=mode)
        if "part" in sub:
            rec["part"] = sub["part"].iloc[0]
        rec.update(n_cp=len(cp), n_base=len(base))
        r = decide(**kw)
        rec["verdict"] = "CP" if r["use_cp"] else "no-CP"
        rec["lcp"] = r["lcp"]

        if len(cp):
            grid = sorted(int(s) for s in cp["L_cp"].unique())
            on_grid = decide(candidates=grid, margin=0.0,
                             **{k: v for k, v in kw.items()})
            S = on_grid["best_cp_lcp"]
            at_model = float(cp[cp["L_cp"] == S][total_col].iloc[0])
            best = float(cp[total_col].min())
            rec.update(
                lcp_model=S,
                lcp_meas=int(cp.loc[cp[total_col].idxmin(), "L_cp"]),
                regret_pct=(at_model / best - 1.0) * 100)
        else:
            rec.update(lcp_model=None, lcp_meas=None, regret_pct=float("nan"))

        if len(base):
            t_base = nocp_measured(base, mode)
            rec["base_err_pct"] = (r["nocp_pred_ms"] / t_base - 1.0) * 100
            rec["verdict_meas"] = ("CP" if len(cp)
                                   and float(cp[total_col].min()) < t_base
                                   else "no-CP")
        else:
            rec["base_err_pct"] = float("nan")
            rec["verdict_meas"] = None
        out.append(rec)
    return pd.DataFrame(out)
