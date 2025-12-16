from __future__ import annotations

import numpy as np

from macrotones.backtest.engine import run_backtest


def test_reproducibility(backtest_fixture) -> None:
    preds, ff, factors, cfg_port, cfg_costs, cfg = backtest_fixture
    result1 = run_backtest(preds, ff, factors, cfg_port, cfg_costs, cfg)
    result2 = run_backtest(preds, ff, factors, cfg_port, cfg_costs, cfg)
    np.testing.assert_allclose(
        result1["pnl_net"].values,
        result2["pnl_net"].values,
    )
