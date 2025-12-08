from __future__ import annotations

import pandas as pd


def smooth_lambda(series: pd.Series, span: int = 6) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def inverse_variance(series: pd.Series, window: int = 6) -> pd.Series:
    var = series.rolling(window, min_periods=2).var().bfill()
    inv = 1.0 / (var + 1e-4)
    return inv / inv.max()


def macro_lambda_drift(
    lambda_series: pd.Series,
    macro_proxy: pd.Series,
    window: int = 6,
) -> pd.Series:
    """
    Compare λ(t) to macro-implied inverse-variance weights.
    """

    lam = lambda_series.astype(float).copy()
    lam.index = pd.to_datetime(lam.index)
    macro = macro_proxy.astype(float).copy()
    macro.index = pd.to_datetime(macro.index)
    macro_norm = inverse_variance(macro, window=window).reindex(lam.index)
    drift = lam - macro_norm
    return drift.dropna()


__all__ = ["inverse_variance", "macro_lambda_drift", "smooth_lambda"]
