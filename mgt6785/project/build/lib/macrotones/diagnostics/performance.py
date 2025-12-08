from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _concat_series(series: Sequence[pd.Series]) -> pd.DataFrame:
    frames = [s.rename(i) for i, s in enumerate(series)]
    df = pd.concat(frames, axis=1)
    df.columns = [f"c{i}" for i in range(len(frames))]
    df.index = pd.to_datetime(df.index)
    return df


def rolling_factor_regression(
    strategy: pd.Series,
    factors: pd.DataFrame,
    window: int = 36,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """
    Estimate rolling alpha/betas via OLS using monthly returns.
    """

    if window <= 1:
        raise ValueError("window must be > 1 for rolling regression.")
    min_obs = min_obs or max(12, window // 2)
    strategy = _ensure_datetime_index(strategy.astype(float))
    factors = factors.copy()
    factors.index = pd.to_datetime(factors.index)
    merged = factors.reindex(strategy.index)
    merged.insert(0, "strategy", strategy)
    merged = merged.dropna(subset=["strategy"]).dropna(how="all")
    if merged.empty:
        return pd.DataFrame()

    cols = [col for col in merged.columns if col != "strategy"]
    if not cols:
        return pd.DataFrame()

    rows: list[dict[str, float | pd.Timestamp]] = []
    for end in range(window - 1, len(merged)):
        window_df = merged.iloc[end - window + 1 : end + 1].dropna()
        if len(window_df) < min_obs:
            continue
        y = window_df["strategy"].to_numpy()
        X = window_df[cols].to_numpy()
        X_design = np.column_stack([np.ones(len(window_df)), X])
        try:
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals = y - X_design @ beta
        resid_vol = float(np.std(residuals, ddof=0))
        window_mean = window_df[cols].mean()
        row: dict[str, float | pd.Timestamp] = {
            "date": window_df.index[-1],
            "alpha": float(beta[0]),
            "residual_vol": resid_vol,
        }
        for idx, factor in enumerate(cols, start=1):
            coeff = float(beta[idx])
            contrib = float(coeff * window_mean[factor])
            row[f"beta_{factor}"] = coeff
            row[f"contrib_{factor}"] = contrib
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def rolling_tracking_error(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int = 12,
    min_obs: int = 6,
) -> pd.DataFrame:
    strategy = _ensure_datetime_index(strategy.astype(float))
    benchmark = _ensure_datetime_index(benchmark.astype(float)).reindex(strategy.index)
    df = pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).dropna()
    if df.empty:
        return pd.DataFrame()
    diff = df["strategy"] - df["benchmark"]
    roll_std = diff.rolling(window, min_periods=min_obs).std(ddof=0)
    roll_mean = diff.rolling(window, min_periods=min_obs).mean()
    info = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan) * np.sqrt(
        MONTHS_PER_YEAR
    )
    corr = df["strategy"].rolling(window, min_periods=min_obs).corr(df["benchmark"])
    return pd.DataFrame(
        {
            "tracking_error": roll_std,
            "information_ratio": info,
            "rolling_corr": corr,
        }
    ).dropna(how="all")


def excess_market_series(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    strategy = _ensure_datetime_index(strategy.astype(float))
    benchmark = _ensure_datetime_index(benchmark.astype(float)).reindex(strategy.index)
    df = pd.DataFrame(
        {"strategy": strategy, "benchmark": benchmark},
        index=strategy.index,
    )
    df["excess_vs_market"] = df["strategy"] - df["benchmark"]
    return df


@dataclass(slots=True)
class DrawdownPeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    depth: float


def _drawdown_periods(ret: pd.Series) -> list[DrawdownPeriod]:
    ret = _ensure_datetime_index(ret.astype(float))
    equity = (1 + ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    periods: list[DrawdownPeriod] = []
    in_dd = False
    start = equity.index[0] if not equity.empty else None
    trough = 0.0

    for date, value in drawdown.items():
        if value < 0 and not in_dd:
            in_dd = True
            start = date
            trough = value
        elif value < 0 and in_dd:
            if value < trough:
                trough = value
        elif value >= -1e-9 and in_dd and start is not None:
            periods.append(DrawdownPeriod(start=start, end=date, depth=float(trough)))
            in_dd = False
            start = None
            trough = 0.0

    if in_dd and start is not None:
        periods.append(
            DrawdownPeriod(start=start, end=ret.index[-1], depth=float(trough))
        )
    periods.sort(key=lambda p: p.depth)
    return periods


def drawdown_attribution(
    bt: pd.DataFrame,
    strategy: pd.Series,
    top_n: int = 3,
) -> pd.DataFrame:
    if bt.empty or strategy.empty:
        return pd.DataFrame()
    weight_cols = [col for col in bt.columns if col.startswith("w_")]
    ret_cols = [col for col in bt.columns if col.startswith("ret_")]
    if not weight_cols or not ret_cols:
        return pd.DataFrame()
    weights = bt[weight_cols].copy()
    returns = bt[ret_cols].copy()
    weights.index = pd.to_datetime(weights.index)
    returns.index = pd.to_datetime(returns.index)
    weights = weights.reindex(strategy.index).fillna(0.0)
    returns = returns.reindex(strategy.index).fillna(0.0)
    weights = weights.rename(columns=lambda col: col.replace("w_", ""))
    returns = returns.rename(columns=lambda col: col.replace("ret_", ""))
    common = sorted(set(weights.columns) & set(returns.columns))
    if not common:
        return pd.DataFrame()
    contrib = weights[common].shift(1).mul(returns[common], axis=0)
    periods = _drawdown_periods(strategy)
    if not periods:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for period in periods[:top_n]:
        mask = (contrib.index >= period.start) & (contrib.index <= period.end)
        window = contrib.loc[mask]
        if window.empty:
            continue
        totals = window.sum()
        for factor, pnl in totals.items():
            rows.append(
                {
                    "event": f"{period.start:%Y-%m}",
                    "start": period.start,
                    "end": period.end,
                    "depth": period.depth,
                    "factor": factor,
                    "contribution": float(pnl),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_sharpe_distribution(
    strategy: pd.Series,
    rf: pd.Series | float | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    strategy = _ensure_datetime_index(strategy.astype(float))
    if isinstance(rf, pd.Series):
        rf_aligned = _ensure_datetime_index(rf.astype(float)).reindex(strategy.index)
        excess = strategy - rf_aligned.fillna(0.0)
    else:
        rf_value = float(rf or 0.0)
        excess = strategy - rf_value
    excess = excess.dropna()
    n = len(excess)
    if n < 3:
        return {"n": n, "point_estimate": float("nan")}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    values = excess.to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        std = sample.std(ddof=0)
        if std <= 1e-12:
            continue
        sr = (sample.mean() / std) * np.sqrt(MONTHS_PER_YEAR)
        samples.append(float(sr))
    sample_array = np.array(samples)
    base_std = excess.std(ddof=0)
    base_sharpe = (
        float((excess.mean() / base_std) * np.sqrt(MONTHS_PER_YEAR))
        if base_std > 1e-12
        else float("nan")
    )
    if base_std > 1e-12:
        base_t = float(excess.mean() / (base_std / np.sqrt(n)))
    else:
        base_t = float("nan")
    p_value = (
        float(2 * stats.t.sf(abs(base_t), df=n - 1))
        if np.isfinite(base_t)
        else float("nan")
    )
    summary = {
        "n": int(n),
        "point_estimate": base_sharpe,
        "t_stat": base_t,
        "p_value": p_value,
        "bootstrap_mean": float(sample_array.mean()) if samples else float("nan"),
        "bootstrap_std": float(sample_array.std(ddof=0)) if samples else float("nan"),
        "p05": float(np.nanpercentile(sample_array, 5)) if samples else float("nan"),
        "p50": float(np.nanpercentile(sample_array, 50)) if samples else float("nan"),
        "p95": float(np.nanpercentile(sample_array, 95)) if samples else float("nan"),
    }
    return summary


def assign_regimes(
    regime_series: pd.Series,
    quantiles: tuple[float, float] = (0.3, 0.7),
) -> pd.Series:
    regime_series = _ensure_datetime_index(regime_series.astype(float)).dropna()
    if regime_series.empty:
        return pd.Series(dtype="object")
    q_low, q_high = regime_series.quantile(list(quantiles))
    labels = pd.Series(index=regime_series.index, dtype="object")
    labels[regime_series <= q_low] = "Risk-Off"
    labels[(regime_series > q_low) & (regime_series < q_high)] = "Neutral"
    labels[regime_series >= q_high] = "Risk-On"
    return labels


def regime_information_table(
    strategy: pd.Series,
    benchmark: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame()
    strategy = _ensure_datetime_index(strategy.astype(float))
    benchmark = _ensure_datetime_index(benchmark.astype(float)).reindex(strategy.index)
    regimes = _ensure_datetime_index(regimes)
    if regimes.index.has_duplicates:
        regimes = regimes[~regimes.index.duplicated(keep="last")]
    regimes = regimes.reindex(strategy.index)
    df = pd.DataFrame(
        {"strategy": strategy, "benchmark": benchmark, "regime": regimes}
    ).dropna()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for regime, group in df.groupby("regime"):
        diff = group["strategy"] - group["benchmark"]
        te = float(diff.std(ddof=0))
        corr = float(group["strategy"].corr(group["benchmark"]))
        info = (
            float((diff.mean() / te) * np.sqrt(MONTHS_PER_YEAR))
            if te > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "regime": regime,
                "information_ratio": info,
                "tracking_error": te,
                "correlation": corr,
                "n_obs": len(group),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


__all__ = [
    "assign_regimes",
    "bootstrap_sharpe_distribution",
    "drawdown_attribution",
    "excess_market_series",
    "regime_information_table",
    "rolling_factor_regression",
    "rolling_tracking_error",
]
