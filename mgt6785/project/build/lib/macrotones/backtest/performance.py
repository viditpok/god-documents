from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from macrotones.config.schema import ConfigModel
from macrotones.datahub import cache_regime_states
from macrotones.diagnostics.performance import (
    assign_regimes,
    bootstrap_sharpe_distribution,
    drawdown_attribution,
    excess_market_series,
    regime_information_table,
    rolling_factor_regression,
    rolling_tracking_error,
)
from macrotones.utils.seed import DEFAULT_SEED

SERIES_CANDIDATES = ["net_ret", "port_ret", "strategy"]
BENCHMARK_CANDIDATES = ["benchmark_ret", "Mkt_RF", "market", "mkt_rf"]
FACTOR_COLUMNS = ["Mkt_RF", "SMB", "HML", "UMD", "LMD"]


def _pick_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype(float)
    raise KeyError(f"None of the candidate columns {candidates} found.")


def _load_llm_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(log_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")
    return df


def generate_performance_artifacts(
    bt: pd.DataFrame,
    ff: pd.DataFrame,
    processed_dir: Path,
    config: ConfigModel,
) -> None:
    if bt.empty or ff.empty:
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    net = _pick_series(bt, SERIES_CANDIDATES)
    try:
        benchmark = _pick_series(bt, BENCHMARK_CANDIDATES)
    except KeyError:
        benchmark = (
            ff.get("Mkt_RF", pd.Series(index=net.index, dtype=float))
            .reindex(net.index)
            .fillna(0.0)
        )
    rf_series = bt.get(config.project.rf_col) or bt.get("rf")
    if rf_series is None:
        rf = pd.Series(0.0, index=net.index)
    else:
        rf = rf_series.astype(float).reindex(net.index).fillna(0.0)
    factor_cols = [col for col in FACTOR_COLUMNS if col in ff.columns]
    factor_returns = ff[factor_cols].reindex(net.index)

    alpha_df = rolling_factor_regression(net, factor_returns, window=36)
    if not alpha_df.empty:
        alpha_df.to_parquet(processed_dir / "alpha_decomposition.parquet")

    tracking_df = rolling_tracking_error(net, benchmark, window=12)
    if not tracking_df.empty:
        tracking_df.to_parquet(processed_dir / "tracking_error.parquet")

    excess_df = excess_market_series(net, benchmark)
    excess_df.to_parquet(processed_dir / "excess_market.parquet")

    drawdown_df = drawdown_attribution(bt, net, top_n=5)
    if not drawdown_df.empty:
        drawdown_df.to_csv(processed_dir / "drawdown_attribution.csv", index=False)

    bootstrap_summary = bootstrap_sharpe_distribution(net, rf, seed=DEFAULT_SEED)
    (processed_dir / "bootstrap_sharpe.json").write_text(
        json.dumps(bootstrap_summary, indent=2), encoding="utf-8"
    )

    log_path = processed_dir / "llm_policy_log.csv"
    llm_log = _load_llm_log(log_path)
    if not llm_log.empty and "nlp_regime" in llm_log.columns:
        regime_series = llm_log["nlp_regime"]
        regimes = assign_regimes(regime_series)
        regime_table = regime_information_table(net, benchmark, regimes)
        if not regime_table.empty:
            regime_table.to_csv(processed_dir / "regime_information.csv")
        cache_payload = pd.DataFrame(
            {
                "regime_score": regime_series,
                "regime_label": regimes.reindex(regime_series.index),
                "lambda": llm_log.get("lambda"),
                "mri": llm_log.get("mri"),
            }
        ).dropna(how="all")
        if not cache_payload.empty:
            cache_regime_states(cache_payload)


__all__ = ["generate_performance_artifacts"]
