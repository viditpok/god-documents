from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from macrotones.backtest.config import BacktestConfig
from macrotones.backtest.engine import run_backtest


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict, dict]:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    factors = ["HML", "SMB"]
    preds = pd.DataFrame(
        {
            "HML": np.linspace(0.004, 0.002, len(idx)),
            "SMB": np.linspace(0.002, 0.006, len(idx)),
            "RF": np.full(len(idx), 0.2),
        },
        index=idx,
    )
    ff = pd.DataFrame(
        {
            "HML": np.linspace(0.002, 0.004, len(idx)),
            "SMB": np.linspace(-0.001, 0.003, len(idx)),
            "RF": np.full(len(idx), 0.2),
            "Mkt_RF": np.linspace(0.001, 0.002, len(idx)),
        },
        index=idx,
    )
    cfg_port = {
        "allocator": "sharpe",
        "long_only": True,
        "trade_threshold": 0.0,
    }
    cfg_costs = {"model": "fixed", "fixed_bps": 0.0}
    return preds, ff, factors, cfg_port, cfg_costs


def test_net_pnl_matches_transaction_cost(sample_bt_cfg: BacktestConfig) -> None:
    preds, ff, factors, cfg_port, cfg_costs = _sample_frames()
    bt = run_backtest(preds, ff, factors, cfg_port, cfg_costs, sample_bt_cfg)
    np.testing.assert_allclose(
        bt["pnl_net"].values,
        bt["pnl_gross"].values - bt["transaction_cost"].values,
    )


def test_benchmark_alignment_matches_returns_index(
    sample_bt_cfg: BacktestConfig,
) -> None:
    preds, ff, factors, cfg_port, cfg_costs = _sample_frames()
    bt = run_backtest(preds, ff, factors, cfg_port, cfg_costs, sample_bt_cfg)
    benchmark_series = bt["benchmark_ret"]
    assert benchmark_series.index.equals(bt.index)
    assert len(benchmark_series) == len(bt)
    assert not benchmark_series.isna().any()


def test_cost_impact_is_small_with_10bps(sample_bt_cfg: BacktestConfig) -> None:
    preds, ff, factors, cfg_port, cfg_costs = _sample_frames()
    zero_cost_cfg = replace(sample_bt_cfg, cost_bps=0.0)
    high_cost_cfg = replace(sample_bt_cfg, cost_bps=10.0)
    bt_zero = run_backtest(preds, ff, factors, cfg_port, cfg_costs, zero_cost_cfg)
    bt_ten = run_backtest(preds, ff, factors, cfg_port, cfg_costs, high_cost_cfg)
    tr_zero = (1 + bt_zero["net_ret"]).prod() - 1
    tr_ten = (1 + bt_ten["net_ret"]).prod() - 1
    # Allow a tiny cushion above 10 bps to avoid flakiness from numerical rounding.
    assert abs(tr_zero - tr_ten) <= 0.0015
