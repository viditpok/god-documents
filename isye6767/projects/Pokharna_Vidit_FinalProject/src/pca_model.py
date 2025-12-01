"""Principal component calculations for the stat-arb model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    """Container describing top eigen-portfolios for a window."""

    tokens: List[str]
    eigenvalues: np.ndarray
    weight_vectors: Dict[int, pd.Series]
    factor_window: pd.DataFrame
    window_returns: pd.DataFrame
    factor_return_at_time: Dict[int, float]


class PCAFactorModel:
    """Runs PCA on standardized returns."""

    def __init__(self, n_factors: int = 2) -> None:
        self.n_factors = n_factors

    def compute(self, window_returns: pd.DataFrame) -> Optional[PCAResult]:
        """Compute PCA outputs for supplied returns window."""
        if window_returns.empty or window_returns.shape[1] < self.n_factors:
            return None

        clean = window_returns.dropna(axis=0, how="any")
        if clean.shape[0] < self.n_factors + 5:
            return None

        means = clean.mean()
        # Suppress warnings for std calculation with NaN values (expected in financial data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stds = clean.std(ddof=1)
        valid = stds > 0
        clean = clean.loc[:, valid]
        means = means[valid]
        stds = stds[valid]
        if clean.shape[1] < self.n_factors:
            return None

        normalized = (clean - means) / stds
        y = normalized.values
        corr = np.corrcoef(y, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(corr)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        tokens = list(clean.columns)
        weights: Dict[int, pd.Series] = {}
        factor_window = pd.DataFrame(index=clean.index)
        factor_at_time: Dict[int, float] = {}
        for idx in range(self.n_factors):
            vec = eigvecs[:, idx]
            weight = pd.Series(vec / stds.values, index=tokens)
            weights[idx + 1] = weight
            factor_values = clean.values @ weight.values
            factor_window[f"F{idx + 1}"] = factor_values
            # Last available return corresponds to most recent hour in window
            factor_at_time[idx + 1] = factor_values[-1]

        return PCAResult(
            tokens=tokens,
            eigenvalues=eigvals[: self.n_factors],
            weight_vectors=weights,
            factor_window=factor_window,
            window_returns=clean,
            factor_return_at_time=factor_at_time,
        )

