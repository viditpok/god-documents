from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from macrotones.config.schema import ConfigModel

BenchmarkName = Literal["SPY", "EQUAL_FF", "RISK_PARITY"]


@dataclass(slots=True)
class BacktestConfig:
    rebalance_freq: str = "M"
    trade_eps: float = 0.0
    borrow_cap: float | None = None
    floor_cash: float | None = None
    rf_col: str = "RF"
    beta_neutral: bool = False
    risk_halflife: int = 12
    cost_bps: float = 10.0
    benchmark: BenchmarkName = "SPY"

    def model_dump(self) -> dict[str, object]:
        return {
            "rebalance_freq": self.rebalance_freq,
            "trade_eps": self.trade_eps,
            "borrow_cap": self.borrow_cap,
            "floor_cash": self.floor_cash,
            "rf_col": self.rf_col,
            "beta_neutral": self.beta_neutral,
            "risk_halflife": self.risk_halflife,
            "cost_bps": self.cost_bps,
            "benchmark": self.benchmark,
        }

    @classmethod
    def from_model(cls, cfg: ConfigModel) -> BacktestConfig:
        port = cfg.portfolio
        project = cfg.project
        return cls(
            rebalance_freq=project.rebalance,
            trade_eps=float(port.trade_threshold),
            borrow_cap=port.borrow_cap,
            floor_cash=port.floor_cash,
            rf_col=project.rf_col,
            beta_neutral=bool(port.beta_neutral),
            risk_halflife=int(port.risk_halflife),
            cost_bps=float(port.cost_bps),
            benchmark=project.benchmark,
        )


__all__ = ["BacktestConfig", "BenchmarkName"]
