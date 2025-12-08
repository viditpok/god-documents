from __future__ import annotations

import numpy as np
import pandas as pd

from macrotones.utils.regime import inverse_variance, smooth_lambda


def test_smooth_lambda_reduces_variance() -> None:
    series = pd.Series([0, 1, 0, 1, 0, 1], dtype=float)
    smoothed = smooth_lambda(series, span=3)
    assert smoothed.var() < series.var()


def test_inverse_variance_positive() -> None:
    series = pd.Series(np.linspace(-1, 1, 12))
    inv = inverse_variance(series)
    assert (inv >= 0).all()
    assert inv.max() == 1.0
