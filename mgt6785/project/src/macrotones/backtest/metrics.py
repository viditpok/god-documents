from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def _clean_series(ret: pd.Series) -> pd.Series:
    return ret.astype(float).dropna()


def annualize_monthly(ret: pd.Series) -> tuple[float, float]:
    clean = _clean_series(ret)
    if clean.empty:
        return float("nan"), float("nan")
    ann_return = float((1 + clean).prod() ** (MONTHS_PER_YEAR / len(clean)) - 1)
    ann_vol = float(clean.std(ddof=0) * np.sqrt(MONTHS_PER_YEAR))
    return ann_return, ann_vol


def sharpe_excess(ret_excess: pd.Series) -> float:
    clean = _clean_series(ret_excess)
    if clean.empty:
        return float("nan")
    sigma = float(clean.std(ddof=0))
    if sigma <= 0 or not np.isfinite(sigma):
        return float("nan")
    excess_mean = float(clean.mean())
    return float((excess_mean / sigma) * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(ret_m: pd.Series) -> float:
    eq = (1 + ret_m).cumprod()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    return float(dd.min())


def summary(
    ret_m: pd.Series,
    ret_excess: pd.Series | None = None,
) -> dict[str, float]:
    ann_return, ann_volatility = annualize_monthly(ret_m)
    excess = ret_m if ret_excess is None else ret_excess.reindex(ret_m.index)
    return {
        "AnnRet": ann_return,
        "AnnVol": ann_volatility,
        "Sharpe_excess": sharpe_excess(excess),
        "MaxDD": max_drawdown(ret_m),
    }
