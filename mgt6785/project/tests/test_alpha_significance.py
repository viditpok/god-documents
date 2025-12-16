from __future__ import annotations

import numpy as np
import pandas as pd

from macrotones.diagnostics import performance


def _build_sample_series(periods: int = 48) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=periods, freq="ME")
    factors = pd.DataFrame(
        {
            "Mkt_RF": rng.normal(0.005, 0.03, size=periods),
            "HML": rng.normal(0.002, 0.02, size=periods),
            "SMB": rng.normal(0.001, 0.02, size=periods),
        },
        index=idx,
    )
    beta = np.array([0.8, 0.3, -0.2])
    noise = rng.normal(0.0, 0.01, size=periods)
    strategy = 0.001 + factors.to_numpy() @ beta + noise
    return pd.Series(strategy, index=idx), factors


def test_rolling_factor_regression_outputs_coefficients() -> None:
    strategy, factors = _build_sample_series()
    frame = performance.rolling_factor_regression(strategy, factors, window=24)
    assert not frame.empty
    assert "alpha" in frame.columns
    assert "beta_Mkt_RF" in frame.columns
    assert frame["beta_Mkt_RF"].notna().any()


def test_bootstrap_sharpe_distribution_reports_quantiles() -> None:
    strategy, _ = _build_sample_series()
    summary = performance.bootstrap_sharpe_distribution(strategy, n_boot=256, seed=7)
    assert summary["n"] == len(strategy)
    assert "p05" in summary and "p95" in summary
    assert summary["point_estimate"] == summary["point_estimate"]
