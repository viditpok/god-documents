"""Performance metrics utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 24 * 365) -> float:
    """Compute annualized Sharpe ratio from periodic returns."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    mean = clean.mean()
    std = clean.std(ddof=1)
    if std == 0:
        return float("nan")
    return (mean / std) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the maximum drawdown of an equity curve."""
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    return drawdowns.min()

