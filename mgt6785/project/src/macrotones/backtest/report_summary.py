from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from macrotones.backtest.config import BacktestConfig
from macrotones.config.schema import ConfigModel


@dataclass(slots=True)
class SummaryStrings:
    header: str
    subtitle: str


def _period_text(
    dates: Collection[pd.Timestamp],
    cfg: ConfigModel,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates_list = list(dates)
    if not dates_list:
        return (pd.Timestamp(cfg.project.start), pd.Timestamp(cfg.project.end))
    series = pd.Series(pd.to_datetime(dates_list)).sort_values()
    return series.iloc[0], series.iloc[-1]


def performance_header(
    cfg: ConfigModel,
    bt_cfg: BacktestConfig,
    dates: Collection[pd.Timestamp],
) -> str:
    start, end = _period_text(dates, cfg)
    return (
        f"Benchmark: {cfg.project.benchmark} | "
        f"Costs: {bt_cfg.cost_bps:.0f} bps | "
        f"Period: {start:%Y-%m}-{end:%Y-%m}"
    )


def report_summary(
    cfg: ConfigModel,
    bt_cfg: BacktestConfig | None,
    dates: Collection[pd.Timestamp],
) -> SummaryStrings:
    bt_cfg = bt_cfg or BacktestConfig.from_model(cfg)
    header = performance_header(cfg, bt_cfg, dates)
    subtitle = f"Net of {bt_cfg.cost_bps:.0f} bps per rebalance"
    return SummaryStrings(header=header, subtitle=subtitle)


__all__ = ["SummaryStrings", "performance_header", "report_summary"]
