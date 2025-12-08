from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ols_baseline import OLSBaseline
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
PANEL_DIR = DATA_PROCESSED / "phase_3_fair_value_panel"


def run_ols_baseline(
    start: str | None = None,
    end: str | None = None,
    min_obs: int = 50,
) -> None:
    """
    Baseline OLS peer-implied fair value:

        P_{i,t}  = a_t + b_t' X_{i,t} + e_{i,t}

    where P_{i,t} is market equity (mktcap) and X are winsorized
    accounting vars (*_w). Mispricing:

        M_{i,t} = (P_{i,t} - P̂_{i,t}) / P̂_{i,t}
    """

    design_path = PANEL_DIR / "design_matrix.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"design matrix not found at {design_path}")

    logger.info("loading design matrix from %s", design_path)
    df = pd.read_csv(design_path, parse_dates=["month_end"])

    if start is not None:
        df = df[df["month_end"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["month_end"] <= pd.to_datetime(end)]

    df = df.copy()
    if "mktcap" not in df.columns:
        raise ValueError("design_matrix.csv must contain mktcap column")
    df = df[df["mktcap"].notna()]

    feature_cols = [c for c in df.columns if c.endswith("_w")]
    if not feature_cols:
        raise ValueError("no *_w feature columns found in design matrix")

    logger.info("using %d winsorized features", len(feature_cols))

    all_preds: list[pd.DataFrame] = []

    for month, grp in df.groupby("month_end"):
        grp = grp[grp["mktcap"].notna()].copy()
        X = grp[feature_cols].copy()
        if X.empty:
            logger.info("%s: no usable features", month.date())
            continue
        X = X.fillna(0.0)
        n = len(X)
        if n < min_obs:
            logger.info(
                "skipping %s: only %d obs (min_obs=%d)", month.date(), n, min_obs
            )
            continue

        y = grp["mktcap"]

        model = OLSBaseline().fit(X, y)
        y_hat = model.predict(X)

        mask = y_hat != 0
        grp = grp.loc[mask].copy()
        y = y.loc[mask]
        y_hat = y_hat.loc[mask]
        M = (y_hat - y) / y

        out_chunk = pd.DataFrame(
            {
                "permno": grp["permno"].values,
                "gvkey": grp["gvkey"].values,
                "month_end": grp["month_end"].values,
                "mktcap": y.values,
                "mktcap_hat": y_hat.values,
                "mispricing_ols": M.values,
            }
        )

        all_preds.append(out_chunk)
        logger.info("month %s: n=%d, fitted OLS baseline", month.date(), len(out_chunk))

    if not all_preds:
        raise RuntimeError("no months met min_obs; mispricing file would be empty")

    mispr_panel = pd.concat(all_preds, ignore_index=True)

    out_path = DATA_PROCESSED / "mispricing_ols.csv"
    mispr_panel.to_csv(out_path, index=False)
    logger.info("saved OLS mispricing panel to %s (%d rows)", out_path, len(mispr_panel))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline OLS peer-implied fair value and mispricing."
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--min-obs", type=int, default=50)
    args = parser.parse_args()

    logger.info(
        "running OLS baseline for %s–%s (min_obs=%d)",
        args.start,
        args.end,
        args.min_obs,
    )
    run_ols_baseline(start=args.start, end=args.end, min_obs=args.min_obs)


if __name__ == "__main__":
    main()
