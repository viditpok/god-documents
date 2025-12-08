from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def ew_cov(df: pd.DataFrame, halflife: int = 12) -> pd.DataFrame | None:
    if df is None:
        return None
    frame = df.dropna(how="all")
    if frame.shape[0] < 2:
        return None
    numeric = frame.astype(float)
    cov_all = numeric.ewm(
        halflife=halflife,
        adjust=False,
        ignore_na=True,
    ).cov()
    last_ts = numeric.index[-1]
    if (last_ts, numeric.columns[0]) not in cov_all.index:
        # fall back to final block if the last timestamp was filtered
        cov_last = cov_all.groupby(level=0).tail(len(numeric.columns))
        cov_last = cov_last.droplevel(0)
    else:
        cov_last = cov_all.loc[last_ts]
    return cov_last.reindex(index=numeric.columns, columns=numeric.columns)


def ledoit_wolf_shrink(cov: np.ndarray) -> np.ndarray:
    mat = np.asarray(cov, dtype=float)
    if mat.size == 0:
        return mat
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("covariance matrix must be square")
    p = mat.shape[0]
    mu = float(np.trace(mat) / max(p, 1))
    target = np.eye(p) * mu
    diff = mat - target
    denom = float(np.sum(diff**2))
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, 0.0)
    numer = float(np.sum(off_diag**2))
    if denom <= EPS:
        shrink = 0.0
    else:
        shrink = float(np.clip(numer / denom, 0.0, 1.0))
    return (1.0 - shrink) * mat + shrink * target


def risk_model(
    returns_df: pd.DataFrame | None,
    halflife: int = 12,
) -> pd.DataFrame | None:
    if returns_df is None:
        return None
    cov = ew_cov(returns_df, halflife=halflife)
    if cov is None or cov.empty:
        return None
    shrunk = ledoit_wolf_shrink(cov.to_numpy())
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


__all__ = ["ew_cov", "ledoit_wolf_shrink", "risk_model"]
