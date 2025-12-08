from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from macrotones.data.macro_loader import get_macro_features
from macrotones.data.nlp_sentiment import get_nlp_regime
from macrotones.fusion.llm_allocator import generate_policy, get_last_rationale


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def _project_beta_neutral(
    weights: pd.Series,
    cov: pd.DataFrame | None,
    market_col: str | None,
) -> pd.Series:
    if market_col is None or cov is None or market_col not in cov.columns:
        return weights
    aligned = cov.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    if market_col not in aligned.columns:
        return weights
    var_market = float(aligned.loc[market_col, market_col])
    if var_market <= 1e-8 or not np.isfinite(var_market):
        return weights
    beta_vec = aligned[market_col] / var_market
    beta_port = float((weights * beta_vec).sum())
    norm = float((beta_vec**2).sum())
    if norm <= 1e-8:
        return weights
    return weights - beta_vec * (beta_port / norm)


def _prepare_llm_weights(
    policy: Mapping[str, float],
    index: pd.Index,
) -> pd.Series:
    series = pd.Series({k: float(v) for k, v in policy.items()}, dtype=float)
    if series.empty or float(series.abs().sum()) == 0.0:
        series = pd.Series(1.0 / len(index), index=index, dtype=float)
    series = series.reindex(index).fillna(0.0)
    total = float(series.sum())
    if abs(total) > 1e-9:
        series = series / total
    return series


def _blend_lambda(macro: Mapping[str, float], nlp: Mapping[str, float]) -> float:
    mri = float(macro.get("mri", 0.0))
    nlp_regime = float(nlp.get("nlp_regime", 0.0))
    if not np.isfinite(mri):
        mri = 0.0
    if not np.isfinite(nlp_regime):
        nlp_regime = 0.0
    mri = abs(mri)
    nlp_regime = abs(nlp_regime)
    lam = 0.5 * _sigmoid(mri) + 0.5 * _sigmoid(nlp_regime)
    return float(np.clip(lam, 0.1, 0.8))


@dataclass(slots=True)
class HybridWeightResult:
    date: pd.Timestamp
    lambda_blend: float
    macro: dict[str, float]
    nlp: dict[str, float]
    w_ridge: pd.Series
    w_llm: pd.Series
    w_final: pd.Series
    rationale: str | None


def blend_with_llm(
    *,
    date: pd.Timestamp,
    w_ridge: pd.Series,
    beta_neutral: bool = False,
    risk_cov: pd.DataFrame | None = None,
    market_factor: str | None = None,
    macro_features: dict[str, float] | None = None,
    nlp_features: dict[str, float] | None = None,
) -> HybridWeightResult:
    ridge = w_ridge.astype(float).reindex(w_ridge.index).fillna(0.0)
    macro = macro_features or get_macro_features(date)
    nlp = nlp_features or get_nlp_regime(date)
    llm_policy = generate_policy(macro, nlp)
    llm_weights = _prepare_llm_weights(llm_policy, ridge.index)
    if beta_neutral:
        llm_weights = _project_beta_neutral(llm_weights, risk_cov, market_factor)
    lam = _blend_lambda(macro, nlp)
    blended = (1.0 - lam) * ridge + lam * llm_weights
    rationale = get_last_rationale()
    return HybridWeightResult(
        date=pd.Timestamp(date),
        lambda_blend=lam,
        macro=dict(macro),
        nlp=dict(nlp),
        w_ridge=ridge,
        w_llm=llm_weights,
        w_final=blended,
        rationale=rationale,
    )
