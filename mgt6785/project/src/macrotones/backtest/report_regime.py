from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from macrotones.utils.seed import DEFAULT_SEED, set_global_seed

CFG: dict[str, Any] = yaml.safe_load(open("config/project.yaml"))
INT = Path(CFG["data"]["out_interim"])
PRO = Path(CFG["data"]["out_processed"])
PRO.mkdir(parents=True, exist_ok=True)

REGIME_LABELS = ["Risk-Off", "Neutral", "Risk-On"]
QUANTILES = (0.30, 0.70)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pick_series(nlp: pd.DataFrame) -> str:
    if "pos_minus_neg_filled" in nlp.columns:
        return "pos_minus_neg_filled"
    if "pos_minus_neg" in nlp.columns:
        return "pos_minus_neg"
    raise KeyError("NLP parquet missing `pos_minus_neg` columns.")


def assign_regimes(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype="object")
    q_low, q_high = valid.quantile(list(QUANTILES))
    regimes = pd.Series(index=series.index, dtype="object")
    regimes[series <= q_low] = REGIME_LABELS[0]
    regimes[(series > q_low) & (series < q_high)] = REGIME_LABELS[1]
    regimes[series >= q_high] = REGIME_LABELS[2]
    return regimes


def summarize_regime(df: pd.DataFrame) -> dict[str, float | int]:
    excess = df["port_ret"] - df["rf"]
    avg_excess = excess.mean()
    vol_excess = excess.std(ddof=0)
    return {
        "avg_excess_ret": float(avg_excess),
        "vol_excess": float(vol_excess),
        "avg_port_ret": float(df["port_ret"].mean()),
        "n_obs": len(df),
        "avg_nlp_cov_docs": (
            float(df["nlp_cov_docs"].mean()) if "nlp_cov_docs" in df else np.nan
        ),
    }


def save_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df["regime"] == label, "port_ret"] for label in REGIME_LABELS]
    ax.boxplot(data, tick_labels=REGIME_LABELS)
    ax.set_title("Strategy Monthly Returns by NLP Regime")
    ax.set_ylabel("Monthly Return")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def select_return_columns(df: pd.DataFrame) -> tuple[str, str]:
    ret_candidates = ["port_ret", "net_ret", "strategy"]
    rf_candidates = ["rf", CFG["project"].get("rf_col", "RF")]
    ret_col = next((col for col in ret_candidates if col in df.columns), None)
    rf_col = next((col for col in rf_candidates if col in df.columns), None)
    if ret_col is None:
        raise KeyError(
            "Backtest results require a return column (net_ret, port_ret, or strategy)."
        )
    if rf_col is None:
        df["rf"] = 0.0
        rf_col = "rf"
    return ret_col, rf_col


def main() -> None:
    set_global_seed(DEFAULT_SEED)
    configure_logging()
    bt = pd.read_parquet(PRO / "backtest.parquet")
    nlp = pd.read_parquet(INT / "nlp_regime_scores.parquet")
    series_name = pick_series(nlp)
    join_cols = [series_name, "pos_minus_neg", "cov_docs", "pos_minus_neg_filled"]
    join_cols = [c for c in join_cols if c in nlp.columns]
    nlp_sel = nlp[join_cols].copy()
    nlp_sel["nlp_cov_docs"] = nlp_sel.get("cov_docs", np.nan)

    df = bt.join(nlp_sel, how="left")
    ret_col, rf_col = select_return_columns(df)
    df["port_ret"] = df[ret_col]
    df["rf"] = df[rf_col]
    series = df[series_name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = df.dropna(subset=["port_ret", "rf"])
    df[series_name] = series
    available = int(series.notna().sum())
    if available < 12:
        logging.warning("Only %d NLP months available; proceeding anyway.", available)
    elif available < 36:
        logging.warning(
            "Limited NLP coverage (%d months) - results may be noisy.", available
        )

    df["regime"] = assign_regimes(series)
    df["nlp_cov_docs"] = df["nlp_cov_docs"].fillna(0)
    summary: dict[str, dict[str, float | int]] = {}
    for label in REGIME_LABELS:
        subset = df[df["regime"] == label]
        if subset.empty:
            continue
        summary[label] = summarize_regime(subset)
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "regime"
    summary_df = summary_df.sort_index()
    out_csv = PRO / "regime_summary.csv"
    summary_df.to_csv(out_csv)

    out_png = PRO / "regime_boxplot.png"
    if df["regime"].notna().any():
        save_plot(df, out_png)
    else:
        logging.warning("No regime assignments; skipping boxplot.")

    logging.info("Saved regime summary -> %s", out_csv)
    logging.info("Saved regime boxplot -> %s", out_png)
    print(f"Saved regime summary -> {out_csv}")
    print(f"Saved regime boxplot -> {out_png}")

    coverage_path = PRO / "regime_coverage.csv"
    doc_cache_path = INT / "nlp_doc_scores.parquet"
    if doc_cache_path.exists():
        doc_df = pd.read_parquet(doc_cache_path).copy()
        if "doc_month" in doc_df.columns:
            doc_df["doc_month"] = pd.to_datetime(doc_df["doc_month"])
            if "token_limit_hit" not in doc_df.columns:
                if "n_chunks" in doc_df.columns:
                    doc_df["token_limit_hit"] = doc_df["n_chunks"].fillna(0) > 1
                else:
                    doc_df["token_limit_hit"] = False
            grouped = doc_df.groupby(doc_df["doc_month"].dt.to_period("M"))
            coverage = grouped.size().rename("docs").to_frame()
            if "n_chunks" in doc_df.columns:
                coverage["avg_chunks"] = grouped["n_chunks"].mean().fillna(0)
            else:
                coverage["avg_chunks"] = 0.0
            coverage["token_limit_hits"] = grouped["token_limit_hit"].sum()
            coverage = coverage.reset_index(names="doc_month")
            coverage["doc_month"] = coverage["doc_month"].dt.to_timestamp("M")
            coverage["token_hit_pct"] = coverage["token_limit_hits"] / coverage[
                "docs"
            ].replace(0, np.nan)
            coverage.to_csv(coverage_path, index=False)
            logging.info("Saved regime coverage -> %s", coverage_path)
    else:
        logging.info("Doc cache not found; skipping coverage table.")


if __name__ == "__main__":
    main()
