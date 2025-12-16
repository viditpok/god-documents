from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from macrotones.diagnostics.ic import ic_summary
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed


def main() -> None:
    set_global_seed(DEFAULT_SEED)
    cfg: dict[str, Any] = yaml.safe_load(open("config/project.yaml"))
    pro = Path(cfg["data"]["out_processed"])
    raw = Path(cfg["data"]["out_raw"])

    preds = pd.read_parquet(pro / "preds.parquet")
    ff = pd.read_parquet(raw / "ff" / "ff_monthly.parquet")

    preds.index = pd.to_datetime(preds.index)
    ff.index = pd.to_datetime(ff.index)

    FACTORS = [
        c
        for c in ["HML", "SMB", "UMD", "Mkt_RF"]
        if c in preds.columns and c in ff.columns
    ]
    # Exclude Mkt_RF from reported ICs by default:
    REPORT_FACTORS = [f for f in FACTORS if f != "Mkt_RF"]

    if not REPORT_FACTORS:
        print("No overlapping factor columns for diagnostics.")
        return

    # Create realized next-month series (t -> t+1 alignment)
    real_next = ff[FACTORS].shift(-1)
    real_next = real_next.reindex(preds.index)

    # Full-sample IC/hit-rate
    df_sum = ic_summary(preds, real_next, REPORT_FACTORS).sort_index()
    df_sum.to_csv(pro / "ic_summary.csv")

    # Rolling 36m Pearson IC
    roll = {}
    for f in REPORT_FACTORS:
        x = preds[f].astype(float)
        y = real_next[f].astype(float).reindex_like(x)
        roll[f] = x.rolling(36).corr(y)

    pd.DataFrame(roll).to_parquet(pro / "ic_rolling.parquet")

    print("Saved IC diagnostics to:", pro)


if __name__ == "__main__":
    main()
