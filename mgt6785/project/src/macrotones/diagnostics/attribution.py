from __future__ import annotations

import pandas as pd


def factor_attribution(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    if returns.empty or weights.empty:
        return pd.DataFrame(columns=["sum", "std", "Sharpe"])

    aligned_returns = returns.copy()
    aligned_returns.index = pd.to_datetime(aligned_returns.index)
    aligned_weights = weights.copy()
    aligned_weights.index = pd.to_datetime(aligned_weights.index)
    aligned_returns = aligned_returns.reindex(aligned_weights.index).fillna(0.0)
    lagged_weights = aligned_weights.shift(1).ffill().fillna(0.0)
    contrib = aligned_returns.mul(
        lagged_weights.reindex(columns=aligned_returns.columns, fill_value=0.0)
    ).sum(axis=1)
    if contrib.empty:
        return pd.DataFrame(columns=["sum", "std", "Sharpe"])
    yearly = contrib.groupby(contrib.index.year).agg(["sum", "std"])
    yearly["Sharpe"] = yearly["sum"] / yearly["std"].replace(0.0, pd.NA)
    return yearly.fillna(0.0)


__all__ = ["factor_attribution"]
