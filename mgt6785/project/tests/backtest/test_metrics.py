from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrotones.backtest import metrics


def test_annualize_monthly_constant_series() -> None:
    series = pd.Series([0.01] * 12)
    ann_return, ann_vol = metrics.annualize_monthly(series)
    expected_ret = (1.01**12) - 1
    assert abs(ann_return - expected_ret) < 1e-9
    assert ann_vol == pytest.approx(0.0, abs=1e-12)


def test_sharpe_excess_matches_manual() -> None:
    excess = pd.Series([0.01, 0.0, -0.01, 0.02])
    sharpe = metrics.sharpe_excess(excess)
    manual = (excess.mean() / excess.std(ddof=0)) * np.sqrt(12)
    assert abs(sharpe - manual) < 1e-10
