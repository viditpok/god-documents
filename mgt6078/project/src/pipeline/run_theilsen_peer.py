from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.theilsen_peer import TheilSenPeerRegressor
from src.pipeline.backtest import compute_quintile_spreads
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
PANEL_DIR = DATA_PROCESSED / "phase_3_fair_value_panel"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def run_theilsen_peer(
    start: str | None = None,
    end: str | None = None,
    min_obs: int = 5,
    max_subpopulation: float = 1e4,
    n_subsamples: int | None = None,
    n_jobs: int | None = None,
    random_state: int | None = 0,
) -> None:
    """
    Theil–Sen peer regression variant (notes/theilsen_plan.md).

    Saves mispricing estimates to data_processed/mispricing_theilsen.csv
    and monthly coefficient snapshots to results/tables/theilsen_coefficients.csv.
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

    logger.info("running Theil–Sen with %d winsorized features", len(feature_cols))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_preds: list[pd.DataFrame] = []
    coeff_rows: list[pd.Series] = []
    diag_rows: list[dict[str, object]] = []

    for month, grp in df.groupby("month_end"):
        grp = grp.dropna(subset=["mktcap"] + feature_cols)
        n = len(grp)
        if n < min_obs:
            logger.info(
                "skipping %s: only %d obs (min_obs=%d)", month.date(), n, min_obs
            )
            continue

        X = grp[feature_cols]
        y = grp["mktcap"]

        model = TheilSenPeerRegressor(
            max_subpopulation=max_subpopulation,
            n_subsamples=n_subsamples,
            n_jobs=n_jobs,
            random_state=random_state,
        ).fit(X, y)
        y_hat = model.predict(X)

        mask = y_hat != 0
        grp = grp.loc[mask].copy()
        y = y.loc[mask]
        y_hat = y_hat.loc[mask]
        M = (y - y_hat) / y_hat

        out_chunk = pd.DataFrame(
            {
                "permno": grp["permno"].values,
                "gvkey": grp["gvkey"].values,
                "month_end": grp["month_end"].values,
                "mktcap": y.values,
                "mktcap_hat": y_hat.values,
                "mispricing_theilsen": M.values,
            }
        )
        all_preds.append(out_chunk)

        coef_series = model.coef_
        coef_series["intercept"] = model.intercept_
        coef_series["month_end"] = month
        coeff_rows.append(coef_series)

        resid = y - y_hat
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        mad_m = float((M - M.median()).abs().median())
        coef_disp = float(coef_series.drop(labels=["intercept", "month_end"]).std(ddof=0))
        diag_rows.append(
            {
                "month_end": month,
                "n_obs": len(out_chunk),
                "r2": r2,
                "median_mispricing": float(M.median()),
                "mad_mispricing": mad_m,
                "coef_dispersion": coef_disp,
            }
        )

        logger.info(
            "month %s: n=%d, fitted Theil–Sen regression",
            month.date(),
            len(out_chunk),
        )

    if not all_preds:
        raise RuntimeError("no months met min_obs; mispricing file would be empty")

    mispr_panel = pd.concat(all_preds, ignore_index=True)
    coeff_panel = pd.DataFrame(coeff_rows)
    diag_panel = pd.DataFrame(diag_rows)

    mispr_path = DATA_PROCESSED / "mispricing_theilsen.csv"
    mispr_ts_path = DATA_PROCESSED / "mispricing_ts.csv"
    coeff_path = TABLES_DIR / "theilsen_coefficients.csv"
    diag_path = TABLES_DIR / "theilsen_diagnostics.csv"

    mispr_panel.to_csv(mispr_path, index=False)
    mispr_panel.rename(
        columns={"mispricing_theilsen": "mispricing_ts"}
    ).to_csv(mispr_ts_path, index=False)
    coeff_panel.to_csv(coeff_path, index=False)
    diag_panel.to_csv(diag_path, index=False)

    logger.info(
        "saved Theil–Sen mispricing panel to %s (%d rows)",
        mispr_path,
        len(mispr_panel),
    )
    logger.info(
        "saved coefficient snapshots to %s (%d months)",
        coeff_path,
        len(coeff_panel),
    )
    logger.info(
        "saved diagnostics to %s (avg n=%.1f, avg R2=%.3f)",
        diag_path,
        diag_panel["n_obs"].mean(),
        diag_panel["r2"].mean(),
    )

    _generate_quintile_backtest(mispr_panel)


def _generate_quintile_backtest(mispr_panel: pd.DataFrame) -> None:
    full_panel_path = PANEL_DIR / "full_panel.csv"
    if not full_panel_path.exists():
        logger.warning("full_panel.csv not found; skipping quintile backtest")
        return

    panel = pd.read_csv(full_panel_path, parse_dates=["date"])
    merged = mispr_panel.merge(
        panel[["permno", "date", "ret_t1"]],
        left_on=["permno", "month_end"],
        right_on=["permno", "date"],
        how="left",
    )
    merged = merged.dropna(subset=["ret_t1"]).copy()
    if merged.empty:
        logger.warning("no overlapping returns for backtest; skipping")
        return

    quintile_returns = compute_quintile_spreads(
        merged,
        signal_col="mispricing_theilsen",
        return_col="ret_t1",
        date_col="month_end",
    )
    quintile_returns["q1_minus_q5"] = quintile_returns["1"] - quintile_returns["5"]

    backtest_path = TABLES_DIR / "theilsen_quintile_returns.csv"
    quintile_returns.to_csv(backtest_path, index=False)

    avg_spread = quintile_returns["q1_minus_q5"].mean()
    ann_spread = avg_spread * 12
    spread_std = quintile_returns["q1_minus_q5"].std()
    sharpe = (avg_spread / spread_std * np.sqrt(12)) if spread_std > 0 else np.nan

    summary = pd.DataFrame(
        [
            {
                "avg_q1": quintile_returns["1"].mean(),
                "avg_q5": quintile_returns["5"].mean(),
                "avg_q1_minus_q5": avg_spread,
                "ann_q1_minus_q5": ann_spread,
                "q1_minus_q5_sharpe": sharpe,
                "num_months": len(quintile_returns),
            }
        ]
    )
    summary_path = TABLES_DIR / "theilsen_quintile_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info(
        "quintile backtest saved to %s (avg q1-q5=%.3f%%, Sharpe=%.2f)",
        backtest_path,
        avg_spread * 100,
        sharpe,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Theil–Sen peer-implied fair value and mispricing."
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--min-obs", type=int, default=5)
    parser.add_argument("--max-subpopulation", type=float, default=1e4)
    parser.add_argument("--n-subsamples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    logger.info(
        "running Theil–Sen peer regression for %s–%s (min_obs=%d)",
        args.start,
        args.end,
        args.min_obs,
    )

    run_theilsen_peer(
        start=args.start,
        end=args.end,
        min_obs=args.min_obs,
        max_subpopulation=args.max_subpopulation,
        n_subsamples=args.n_subsamples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
