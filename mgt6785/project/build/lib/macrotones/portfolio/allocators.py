from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _safe_series(x: object, index: pd.Index) -> pd.Series:
    s = pd.Series(x, index=index)
    return s.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def get_vol(hist_df: pd.DataFrame | None, window: int = 12) -> pd.Series:
    if hist_df is None or hist_df.empty:
        return pd.Series(np.nan, index=[])
    vol = hist_df.rolling(window).std().iloc[-1]
    return vol


def sharpe_score(
    pred_mu: pd.Series,
    hist_df: pd.DataFrame | None,
    long_only: bool = True,
    window: int = 12,
) -> pd.Series:
    vol = get_vol(hist_df, window).reindex(pred_mu.index)
    vol = vol.replace(0, np.nan)
    score = pred_mu / vol
    if long_only:
        score = score.clip(lower=0)
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def proportional_weights(score: pd.Series, long_only: bool = True) -> pd.Series:
    s = score.copy()
    if long_only:
        s = s.clip(lower=0)
    denom = s.abs().sum()
    if not np.isfinite(denom) or denom <= 0:
        return _safe_series(0.0, s.index)
    return (s / denom).fillna(0.0)


def topk_weights(score: pd.Series, k: int = 2, long_only: bool = True) -> pd.Series:
    s = score.copy()
    if long_only:
        s = s.clip(lower=0)
    keep = s.sort_values(ascending=False).head(k)
    if keep.abs().sum() <= 0:
        return _safe_series(0.0, s.index)
    w = pd.Series(0.0, index=s.index)
    w.loc[keep.index] = keep / keep.abs().sum()
    return w


def softmax_weights(
    score: pd.Series,
    temperature: float = 1.0,
    k_top: int | None = None,
    long_only: bool = True,
) -> pd.Series:
    s = score.copy()
    if long_only:
        s = s.clip(lower=0)
    if k_top is not None and k_top > 0:
        keep = set(s.nlargest(k_top).index)
        s.loc[~s.index.isin(keep)] = -np.inf
    t = max(1e-6, float(temperature))
    # subtract max for numerical stability
    a = np.exp((s - np.nanmax(s)) / t)
    a = a.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    denom = a.sum()
    if denom <= 0:
        return _safe_series(0.0, s.index)
    return (a / denom).reindex(s.index).fillna(0.0)


def trailing_cov(
    returns_df: pd.DataFrame | None, window: int = 36
) -> pd.DataFrame | None:
    if returns_df is None or returns_df.empty:
        return None
    tail = returns_df.dropna().tail(window)
    if tail.shape[0] < 2:
        return None
    return tail.cov()


def postprocess_weights(
    w_raw: pd.Series,
    prev_w: pd.Series | None,
    hist_returns: pd.DataFrame | None,
    cfg: Mapping[str, Any],
    cov_override: pd.DataFrame | None = None,
) -> tuple[pd.Series, float]:
    w = w_raw.copy()

    # Smooth weights
    lam = float(cfg.get("smooth_lambda", 0.0))
    if prev_w is not None and 0.0 < lam < 1.0:
        w = (1 - lam) * w + lam * prev_w.reindex(w.index).fillna(0.0)

    # Per-factor cap and renormalize if needed
    cap = float(cfg.get("weight_cap", 0.0))
    if cap > 0:
        w = w.clip(upper=cap)
        s = w.sum()
        if s > 0:
            w = w / s

    # Compute turnover vs previous weights
    turnover = 0.0
    if prev_w is not None:
        turnover = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())

    # De-lever to vol target (no lever-up)
    target_ann = float(cfg.get("risk_target_ann", 0.0))
    if target_ann > 0.0:
        if cov_override is not None and not cov_override.empty:
            cov = cov_override.reindex(index=w.index, columns=w.index).fillna(0.0)
        else:
            cov = trailing_cov(hist_returns, window=36)
        if cov is not None:
            vec = w.reindex(cov.columns).fillna(0.0).values
            sigma_m = float(np.sqrt(max(0.0, vec @ cov.values @ vec)))
            target_m = target_ann / np.sqrt(12.0)
            if sigma_m > 1e-8 and sigma_m > target_m:
                w *= target_m / sigma_m

    return w.fillna(0.0), turnover


def build_weights(
    pred_mu: pd.Series,
    hist_df: pd.DataFrame | None,
    cfg: Mapping[str, Any],
    prev_w: pd.Series | None = None,
    risk_cov: pd.DataFrame | None = None,

) -> tuple[pd.Series, float, pd.Series]:
    long_only = bool(cfg.get("long_only", True))
    score = sharpe_score(pred_mu, hist_df, long_only=long_only, window=12)

    score = sharpe_score(pred_mu, hist_df, long_only=long_only, window=12)

    # Cash gate
    if float(score.abs().max()) < float(cfg.get("hold_cash_threshold", 0.0)):
        w_raw = _safe_series(0.0, pred_mu.index)
    else:
        alloc = str(cfg.get("allocator", "softmax")).lower()
        if alloc == "softmax":
            k_top_cfg = cfg.get("k_top")
            k_top = int(k_top_cfg) if isinstance(k_top_cfg, int | float) else None
            w_raw = softmax_weights(
                score,
                temperature=float(cfg.get("temperature", 1.0)),
                k_top=k_top,
                long_only=long_only,
            )
        elif alloc == "topk":
            w_raw = topk_weights(score, k=int(cfg.get("k_top", 2)), long_only=long_only)
        else:
            w_raw = proportional_weights(score, long_only=long_only)

    hist_subset = (
        hist_df.reindex(columns=pred_mu.index) if hist_df is not None else None
    )
    w, turnover = postprocess_weights(
        w_raw,
        prev_w,
        hist_subset,
        cfg,
        cov_override=risk_cov,
    )
    return w, turnover, score
