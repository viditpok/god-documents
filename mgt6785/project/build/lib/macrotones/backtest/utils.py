from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from macrotones.backtest.config import BenchmarkName
from macrotones.config.schema import SlippageModel


def is_rebalance_date(date: pd.Timestamp, frequency: str) -> bool:
    frequency = frequency.upper()
    if frequency in {"M", "MONTHLY"}:
        return True
    if frequency in {"Q", "QUARTERLY"}:
        return date.to_period("Q").to_timestamp(how="end").date() == date.date()
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def apply_trade_threshold(
    new_weights: pd.Series,
    prev_weights: pd.Series,
    epsilon: float,
) -> tuple[pd.Series, pd.Series, bool]:
    delta = new_weights - prev_weights
    if epsilon <= 0:
        mask = pd.Series(True, index=new_weights.index)
        return new_weights, delta, bool((delta != 0).any())
    mask = delta.abs() >= epsilon
    adjusted = prev_weights.copy()
    adjusted.loc[mask] = new_weights.loc[mask]
    delta = adjusted - prev_weights
    return adjusted, delta, bool(mask.any())


def enforce_leverage_constraints(
    weights: pd.Series,
    borrow_cap: float | None,
    cash_floor: float | None,
) -> pd.Series:
    adjusted = weights.copy()
    if borrow_cap is not None:
        max_leverage = 1.0 + float(borrow_cap)
        current = float(np.abs(adjusted).sum())
        if current > max_leverage and current > 0:
            adjusted *= max_leverage / current

    if cash_floor is not None:
        floor = float(cash_floor)
        gross = float(adjusted.sum())
        cash = 1.0 - gross
        if cash < floor and gross != 0:
            target_sum = 1.0 - floor
            adjusted *= target_sum / gross
    return adjusted


def compute_slippage(
    model_cfg: Mapping[str, float | str | None],
    delta_weights: pd.Series,
    hist_returns: pd.DataFrame | None,
) -> float:
    model = SlippageModel(str(model_cfg.get("model", SlippageModel.FIXED)))
    turnover = float(np.abs(delta_weights).sum())

    if turnover == 0:
        return 0.0

    if model == SlippageModel.FIXED:
        fixed_bps = float(model_cfg.get("fixed_bps", 0.0))
        return turnover * fixed_bps * 1e-4

    if model == SlippageModel.VOL:
        k = float(model_cfg.get("vol_k", 0.0))
        if hist_returns is None or hist_returns.empty or k == 0:
            return 0.0
        vol = hist_returns.std(ddof=0).reindex(delta_weights.index).fillna(0.0)
        return float((np.abs(delta_weights) * vol).sum() * k)

    if model == SlippageModel.SPREAD:
        spread_bps = float(model_cfg.get("spread_bps", 0.0))
        return turnover * spread_bps * 1e-4

    return 0.0


def load_benchmark_series(
    name: BenchmarkName,
    ff: pd.DataFrame,
    index: Sequence[pd.Timestamp] | pd.Index | None = None,
) -> pd.Series:
    if ff.empty:
        target_index = (
            pd.Index(index)
            if index is not None
            else pd.Index([], dtype="datetime64[ns]")
        )
        return pd.Series(0.0, index=target_index, dtype=float)

    ff_aligned = ff.copy()
    ff_aligned.index = pd.to_datetime(ff_aligned.index)
    name = (name or "SPY").upper()

    def _rf_series() -> pd.Series:
        if "RF" not in ff_aligned.columns:
            return pd.Series(0.0, index=ff_aligned.index, dtype=float)
        return ff_aligned["RF"].astype(float).fillna(0.0) / 1200.0

    rf_series = _rf_series()
    factor_cols = [
        col for col in ["Mkt_RF", "SMB", "HML", "UMD"] if col in ff_aligned.columns
    ]
    if not factor_cols:
        base = pd.Series(0.0, index=ff_aligned.index, dtype=float)
    else:
        base = ff_aligned[factor_cols].astype(float).fillna(0.0)

    if name == "SPY":
        benchmark = ff_aligned.get(
            "Mkt_RF",
            pd.Series(0.0, index=ff_aligned.index, dtype=float),
        ).fillna(0.0)
    elif name == "EQUAL_FF":
        if factor_cols:
            benchmark = base.mean(axis=1)
        else:
            benchmark = pd.Series(0.0, index=ff_aligned.index, dtype=float)
    elif name == "RISK_PARITY":
        if factor_cols:
            vol = base.std(ddof=0)
            inv_vol = 1.0 / vol.replace(0.0, np.nan)
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if inv_vol.sum() == 0:
                weights = np.full(len(factor_cols), 1.0 / len(factor_cols))
            else:
                weights = inv_vol / inv_vol.sum()
            benchmark = (base * weights.values).sum(axis=1)
        else:
            benchmark = pd.Series(0.0, index=ff_aligned.index, dtype=float)
    else:
        benchmark = ff_aligned.get(
            "Mkt_RF",
            pd.Series(0.0, index=ff_aligned.index, dtype=float),
        ).fillna(0.0)

    series = (benchmark + rf_series).astype(float).fillna(0.0)
    if index is not None:
        target_index = pd.Index(index)
        series = series.reindex(target_index).ffill().fillna(0.0)
    return series


__all__ = [
    "apply_trade_threshold",
    "compute_slippage",
    "enforce_leverage_constraints",
    "is_rebalance_date",
    "load_benchmark_series",
]
