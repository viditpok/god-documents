"""Ornstein-Uhlenbeck parameter estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OUParams:
    """Holds OU model estimates for a single asset."""

    a: float
    b: float
    variance_eta: float
    kappa: float
    m: float
    sigma: float
    sigma_eq: float
    s_score: Optional[float] = None


class OUProcessEstimator:
    """Estimates OU parameters from residual time-series."""

    def __init__(self, delta_t: float = 1 / 8760) -> None:
        self.delta_t = delta_t

    def estimate(self, residuals: np.ndarray) -> Optional[OUParams]:
        """Estimate OU parameters from residual series."""
        if residuals.size < 10:
            return None

        x_series = np.cumsum(residuals)
        x_l = x_series[:-1]
        x_next = x_series[1:]
        if x_l.std() == 0:
            return None

        design = np.column_stack([np.ones_like(x_l), x_l])
        coeffs, _, _, _ = np.linalg.lstsq(design, x_next, rcond=None)
        a, b = coeffs
        eta = x_next - (a + b * x_l)
        if eta.size < 2:
            return None
        variance_eta = float(np.var(eta, ddof=1))
        if variance_eta <= 0:
            return None

        if not (0 < b < 0.999999):
            return None

        kappa = -np.log(b) / self.delta_t
        m = a / (1 - b)
        sigma = np.sqrt(
            variance_eta * 2 * kappa / (1 - np.exp(-2 * kappa * self.delta_t))
        )
        sigma_eq = np.sqrt(variance_eta / (1 - b ** 2))
        return OUParams(
            a=float(a),
            b=float(b),
            variance_eta=variance_eta,
            kappa=float(kappa),
            m=float(m),
            sigma=float(sigma),
            sigma_eq=float(sigma_eq),
        )

    @staticmethod
    def compute_s_score(params: OUParams, avg_a: float) -> Optional[float]:
        """Compute centred s-score using Appendix A approximation."""
        if params.variance_eta <= 0 or params.sigma_eq <= 0 or abs(params.b) >= 1:
            return None
        numerator = -(params.a - avg_a) * np.sqrt(1 - params.b ** 2)
        denom = (1 - params.b) * np.sqrt(params.variance_eta)
        if denom == 0:
            return None
        return float(numerator / denom)

