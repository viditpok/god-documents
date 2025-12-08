from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TARGETS = ["HML", "SMB", "UMD"]


def _load_cfg() -> dict[str, Any]:
    with Path("config/project.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_predictions(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pro = Path(cfg["data"]["out_processed"])
    raw = Path(cfg["data"]["out_raw"])
    preds = pd.read_parquet(pro / "preds.parquet").sort_index()
    ff = pd.read_parquet(raw / "ff" / "ff_monthly.parquet").sort_index()
    bt = pd.read_parquet(pro / "backtest.parquet").sort_index()
    return preds, ff, bt


def _compute_alpha_decay(preds: pd.DataFrame, ff: pd.DataFrame, pro: Path) -> None:
    rows: list[dict[str, Any]] = []
    ff_targets = ff[TARGETS].sort_index()

    for factor in TARGETS:
        pred_series = preds.get(factor)
        if pred_series is None:
            continue
        pred_series = pred_series.dropna()
        for horizon in range(1, 7):
            realized = ff_targets[factor].shift(-horizon).reindex(pred_series.index)
            mask = pred_series.notna() & realized.notna()
            if mask.sum() < 12:
                ic = np.nan
            else:
                ic = float(pred_series[mask].corr(realized[mask], method="pearson"))
            rows.append(
                {
                    "factor": factor,
                    "horizon": horizon,
                    "ic": ic,
                }
            )

    decay_df = pd.DataFrame(rows)
    decay_df.to_csv(pro / "alpha_decay.csv", index=False)


def _compute_attribution(bt: pd.DataFrame, pro: Path) -> None:
    weight_cols = [col for col in bt.columns if col.startswith("w_")]
    if not weight_cols:
        return

    contributions = {}
    for col in weight_cols:
        factor = col.replace("w_", "")
        ret_col = f"ret_{factor}"
        if ret_col not in bt.columns:
            continue
        contrib = bt[col].shift(1) * bt[ret_col]
        contributions[factor] = contrib

    if not contributions:
        return

    contrib_df = pd.DataFrame(contributions).dropna(how="all")
    contrib_df.index.name = "date"
    contrib_long = contrib_df.reset_index().melt(
        "date", var_name="factor", value_name="pnl"
    )
    contrib_long.to_csv(pro / "attribution.csv", index=False)

    summary = (
        contrib_df.agg(["sum", "mean"])
        .T.reset_index()
        .rename(columns={"index": "factor", "sum": "total_pnl", "mean": "avg_pnl"})
    )
    total_pnl = summary["total_pnl"].sum()
    if abs(total_pnl) > 1e-12:
        summary["share"] = summary["total_pnl"] / total_pnl
    else:
        summary["share"] = 0.0
    summary.to_csv(pro / "attribution_summary.csv", index=False)


def main() -> None:
    cfg = _load_cfg()
    pro = Path(cfg["data"]["out_processed"])
    preds, ff, bt = _load_predictions(cfg)

    _compute_alpha_decay(preds, ff, pro)
    _compute_attribution(bt, pro)

    print(f"Saved alpha decay -> {pro / 'alpha_decay.csv'}")
    print(f"Saved attribution -> {pro / 'attribution.csv'}")


if __name__ == "__main__":
    main()
