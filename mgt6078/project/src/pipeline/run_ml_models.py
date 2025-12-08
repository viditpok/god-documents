from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ml_fair_value import LassoFairValue, PLSFairValue, XGBFairValue
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
PANEL_DIR = DATA_PROCESSED / "phase_3_fair_value_panel"


MODEL_REGISTRY = {
    "lasso": ("mispricing_lasso", LassoFairValue),
    "pls": ("mispricing_pls", PLSFairValue),
    "xgb": ("mispricing_xgb", XGBFairValue),
}


def run_ml_models(
    models: list[str],
    start: str | None = None,
    end: str | None = None,
    min_obs: int = 50,
) -> None:
    design_path = PANEL_DIR / "design_matrix.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"design matrix not found at {design_path}")

    missing = [m for m in models if m not in MODEL_REGISTRY]
    if missing:
        raise ValueError(f"unknown models requested: {missing}")

    logger.info(
        "running ML models %s for %s–%s (min_obs=%s)",
        models,
        start,
        end,
        min_obs,
    )

    df = pd.read_csv(design_path, parse_dates=["month_end"])
    if start is not None:
        df = df[df["month_end"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["month_end"] <= pd.to_datetime(end)]

    df = df.copy()
    if "mktcap" not in df.columns:
        raise ValueError("design_matrix.csv must contain mktcap column")

    feature_cols = [c for c in df.columns if c.endswith("_w")]
    if not feature_cols:
        raise ValueError("no *_w feature columns found in design matrix")

    outputs: dict[str, list[pd.DataFrame]] = {name: [] for name in models}

    for month, grp in df.groupby("month_end"):
        grp = grp.dropna(subset=["mktcap"] + feature_cols)
        n = len(grp)
        if n < min_obs:
            logger.info("skipping %s: only %d obs", month.date(), n)
            continue

        X = grp[feature_cols].copy()
        y = grp["mktcap"].copy()

        for name in models:
            col, cls = MODEL_REGISTRY[name]
            model = cls()
            model.fit(X, y)
            preds = model.predict(X)
            mask = preds != 0
            mispricing = (preds[mask]-y[mask]) / y[mask]
            chunk = pd.DataFrame(
                {
                    "permno": grp.loc[mask, "permno"].values,
                    "gvkey": grp.loc[mask, "gvkey"].values,
                    "month_end": grp.loc[mask, "month_end"].values,
                    "mktcap": y[mask].values,
                    "mktcap_hat": preds[mask].values,
                    col: mispricing.values,
                }
            )
            outputs[name].append(chunk)
            logger.info("month %s: %s fitted with %d obs", month.date(), name, len(chunk))

    for name in models:
        col, _ = MODEL_REGISTRY[name]
        if not outputs[name]:
            logger.warning("no outputs generated for %s", name)
            continue
        panel = pd.concat(outputs[name], ignore_index=True)
        out_path = DATA_PROCESSED / f"{col}.csv"
        panel.to_csv(out_path, index=False)
        logger.info("saved %s panel to %s (%d rows)", col, out_path, len(panel))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Lasso/PLS/XGB mispricing models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lasso", "pls", "xgb"],
        help="Subset of models to run (options: lasso, pls, xgb)",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--min-obs", type=int, default=50)
    args = parser.parse_args()

    run_ml_models(
        models=args.models,
        start=args.start,
        end=args.end,
        min_obs=args.min_obs,
    )


if __name__ == "__main__":
    main()
