import numpy as np
import pandas as pd
import pytest

from macrotones.backtest.config import BacktestConfig
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed


@pytest.fixture(autouse=True)
def _reset_seed() -> None:
    """Ensure deterministic seeds for every test."""

    set_global_seed(DEFAULT_SEED)


@pytest.fixture
def sample_bt_cfg() -> BacktestConfig:
    """Reusable backtest config for unit tests."""

    return BacktestConfig(
        rebalance_freq="M",
        trade_eps=0.0,
        rf_col="RF",
        beta_neutral=False,
        risk_halflife=12,
        cost_bps=10.0,
        benchmark="SPY",
    )


@pytest.fixture
def backtest_fixture(sample_bt_cfg: BacktestConfig):
    """Provide deterministic inputs for reproducibility tests."""

    idx = pd.date_range("2019-01-31", periods=5, freq="ME")
    preds = pd.DataFrame(
        {
            "HML": np.linspace(0.01, 0.005, len(idx)),
            "SMB": np.linspace(0.005, 0.009, len(idx)),
            "RF": np.full(len(idx), 0.2),
        },
        index=idx,
    )
    ff = pd.DataFrame(
        {
            "HML": np.linspace(0.002, 0.006, len(idx)),
            "SMB": np.linspace(-0.001, 0.004, len(idx)),
            "RF": np.full(len(idx), 0.2),
            "Mkt_RF": np.linspace(0.001, 0.003, len(idx)),
        },
        index=idx,
    )
    factors = ["HML", "SMB"]
    cfg_port = {"allocator": "sharpe", "long_only": True, "trade_threshold": 0.0}
    cfg_costs = {"model": "fixed", "fixed_bps": 0.0}
    return preds, ff, factors, cfg_port, cfg_costs, sample_bt_cfg
