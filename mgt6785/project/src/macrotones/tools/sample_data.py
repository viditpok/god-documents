from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CLI usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_sample_data() -> None:
    np.random.seed(42)

    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    for path in [raw_dir / "ff", raw_dir / "fomc", interim_dir, processed_dir]:
        path.mkdir(parents=True, exist_ok=True)

    periods = 240  # 20 years of synthetic history
    dates = pd.date_range("2005-01-31", periods=periods, freq="ME")

    # Raw Fama-French factors
    time = np.arange(len(dates))
    cyclical = np.sin(time / 6) * 0.005
    trend = 0.0005 * time / len(dates)
    noise = np.random.normal(0, 0.008, size=(len(dates), 3))
    base_factors = 0.004 + trend[:, None] + cyclical[:, None] + noise
    ff = pd.DataFrame(
        base_factors,
        index=dates,
        columns=["HML", "SMB", "UMD"],
    )
    ff["Mkt_RF"] = 0.01 + trend + np.random.normal(0, 0.012, len(dates))
    ff["RF"] = np.full(len(dates), 0.0015)
    ff.to_parquet(raw_dir / "ff" / "ff_monthly.parquet")

    # Raw FRED sample data
    fred_trend = np.linspace(220, 260, len(dates))
    unrate_noise = np.random.normal(0, 0.1, len(dates))
    dgs10_noise = np.random.normal(0, 0.05, len(dates))
    dgs1_noise = np.random.normal(0, 0.05, len(dates))
    fred = pd.DataFrame(
        {
            "CPI": fred_trend + np.random.normal(0, 0.3, len(dates)),
            "UNRATE": 5.5 + 0.5 * np.sin(time / 9) + unrate_noise,
            "DGS10": 2.0 + 0.2 * np.sin(time / 12) + dgs10_noise,
            "DGS1": 1.0 + 0.15 * np.sin(time / 10) + dgs1_noise,
        },
        index=dates,
    )
    fred.to_parquet(raw_dir / "fred_monthly.parquet")

    # Interim NLP doc cache
    doc_rows = []
    for i, dt in enumerate(dates):
        text = (
            f"FOMC meeting statement {dt:%Y-%m-%d}. "
            "Economic outlook remains stable with moderate growth."
        )
        path = raw_dir / "fomc" / f"fomc_{dt:%Y-%m-%d}_{i}.txt"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        probs = np.random.dirichlet([2.5, 5.0, 1.8])
        doc_rows.append(
            {
                "path": str(path),
                "doc_month": dt,
                "sha1": digest,
                "pos": probs[0],
                "neu": probs[1],
                "neg": probs[2],
                "n_chunks": np.random.randint(1, 4),
                "token_limit_hit": bool(np.random.choice([True, False])),
            }
        )

    doc_scores = pd.DataFrame(doc_rows)
    doc_scores.to_parquet(interim_dir / "nlp_doc_scores.parquet", index=False)

    regime_scores = pd.DataFrame(
        {
            "pos_minus_neg": np.random.normal(0, 0.3, len(dates)),
            "pos_minus_neg_filled": np.random.normal(0, 0.3, len(dates)),
            "cov_docs": np.random.randint(1, 6, len(dates)),
        },
        index=dates,
    )
    regime_scores.to_parquet(interim_dir / "nlp_regime_scores.parquet")

    # Processed panel
    panel = fred.copy()
    panel["TERM_SPREAD"] = panel["DGS10"] - panel["DGS1"]
    panel = panel.join(ff[["HML", "SMB", "UMD", "Mkt_RF", "RF"]])
    panel.to_parquet(processed_dir / "panel.parquet")

    # Predictions
    preds_noise = np.random.normal(0, 0.006, size=(len(dates), 3))
    preds = pd.DataFrame(
        0.5 * ff[["HML", "SMB", "UMD"]].values + preds_noise,
        index=dates,
        columns=["HML", "SMB", "UMD"],
    )
    preds["Mkt_RF"] = ff["Mkt_RF"]
    preds["RF"] = ff["RF"]
    preds.to_parquet(processed_dir / "preds.parquet")

    # Backtest sample
    weights = pd.DataFrame(
        {
            "w_HML": 0.3 + 0.1 * np.sin(time / 8),
            "w_SMB": 0.4 + 0.05 * np.cos(time / 7),
            "w_UMD": 0.3 + 0.05 * np.sin(time / 5 + 0.4),
        },
        index=dates,
    )
    scores = weights.rename(columns=lambda c: c.replace("w_", "s_"))
    net = np.random.normal(0.008, 0.01, len(dates))
    gross = net + np.random.uniform(0.0001, 0.0004, len(dates))
    slippage = gross - net
    backtest = (
        pd.DataFrame(
            {
                "net_ret": net,
                "gross_ret": gross,
                "slippage": slippage,
                "rf": ff["RF"],
                "turnover": np.random.uniform(0.05, 0.2, len(dates)),
            },
            index=dates,
        )
        .join(weights)
        .join(scores)
    )
    backtest.to_parquet(processed_dir / "backtest.parquet")

    # Rolling turnover
    backtest["turnover"].rolling(window=12, min_periods=1).mean().to_frame(
        name="turnover_rolling"
    ).to_parquet(processed_dir / "turnover_rolling.parquet")

    # Summaries
    ablation = pd.DataFrame(
        {
            "factor": ["HML", "SMB", "UMD"],
            "delta_sharpe": [-0.12, -0.06, -0.03],
            "delta_ic": [-0.03, -0.02, -0.01],
        }
    )
    ablation.to_csv(processed_dir / "ablation_summary.csv", index=False)

    sweep = pd.DataFrame(
        {
            "ridge_alpha": [0.5, 1.0, 3.0],
            "temperature": [0.3, 0.5, 0.7],
            "sharpe": [0.9, 1.05, 1.12],
            "ann_ret": [0.10, 0.12, 0.13],
            "ann_vol": [0.09, 0.10, 0.10],
            "max_dd": [-0.12, -0.08, -0.07],
            "is_best": [False, False, True],
        }
    )
    sweep.to_csv(processed_dir / "sweep_summary.csv", index=False)

    regime_summary = pd.DataFrame(
        {
            "regime": ["Neutral", "Risk-Off", "Risk-On"],
            "avg_port_ret": [0.007, -0.004, 0.012],
            "avg_excess_ret": [0.005, -0.006, 0.010],
            "vol_excess": [0.011, 0.009, 0.013],
            "n_obs": [80, 50, 70],
            "avg_nlp_cov_docs": [3.0, 2.4, 3.6],
        }
    ).set_index("regime")
    regime_summary.to_csv(processed_dir / "regime_summary.csv")

    coverage = pd.DataFrame(
        {
            "doc_month": dates,
            "docs": np.random.randint(1, 5, len(dates)),
            "avg_chunks": np.random.uniform(1.0, 2.0, len(dates)),
            "token_limit_hits": np.random.randint(0, 2, len(dates)),
        }
    )
    coverage["token_hit_pct"] = coverage["token_limit_hits"] / coverage["docs"].replace(
        0, np.nan
    )
    coverage.to_csv(processed_dir / "regime_coverage.csv", index=False)

    plt.figure(figsize=(6, 4))
    data = [
        np.random.normal(-0.002, 0.01, 60),
        np.random.normal(0.0, 0.01, 60),
        np.random.normal(0.01, 0.01, 60),
    ]
    plt.boxplot(data, tick_labels=["Risk-Off", "Neutral", "Risk-On"])
    plt.title("Strategy Monthly Returns by NLP Regime")
    plt.ylabel("Monthly Return")
    plt.tight_layout()
    plt.savefig(processed_dir / "regime_boxplot.png", dpi=120)
    plt.close()

    marker = Path("data") / ".sample_seed"
    marker.write_text(
        "sample_data_seeded\n",
        encoding="utf-8",
    )

    print("Sample data written to data/raw, data/interim, and data/processed")


def main() -> None:
    generate_sample_data()


if __name__ == "__main__":
    main()
