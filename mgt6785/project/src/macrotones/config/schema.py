from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)


class AllocatorType(str, Enum):
    SOFTMAX = "softmax"
    TOPK = "topk"
    SHARPE = "sharpe"


class ProjectConfig(BaseModel):
    name: str
    start: date
    end: date
    rebalance: str = Field(min_length=1)
    rf_col: str = Field(default="RF", min_length=1)
    benchmark: Literal["SPY", "EQUAL_FF", "RISK_PARITY"] = "SPY"

    @field_validator("end")
    @classmethod
    def _validate_end(cls, end: date, info) -> date:
        start = info.data.get("start") if info.data else None
        if start and end < start:
            raise ValueError("end date must be on or after project.start")
        return end


class DataConfig(BaseModel):
    fred_series: list[str]
    out_raw: Path
    out_interim: Path
    out_processed: Path


class ModelConfig(BaseModel):
    horizon_months: PositiveInt
    cv_splits: PositiveInt
    min_train_months: PositiveInt
    ensemble: list[str]
    ridge_alpha: PositiveFloat | None = Field(default=None)
    ridge_alpha_grid: list[PositiveFloat] | None = Field(default=None)


class SlippageModel(str, Enum):
    FIXED = "fixed"
    VOL = "vol"
    SPREAD = "spread"


class CostsConfig(BaseModel):
    model: SlippageModel = Field(
        default=SlippageModel.FIXED, alias=AliasChoices("model", "slippage_model")
    )
    fixed_bps: float | None = Field(default=None, ge=0.0)
    vol_k: PositiveFloat | None = Field(default=None)
    spread_bps: float | None = Field(default=None, ge=0.0)
    bps_per_turnover: float | None = Field(
        default=None, ge=0.0, description="Legacy fixed slippage parameter."
    )

    model_config = {"populate_by_name": True}


class PortfolioConfig(BaseModel):
    allocator: AllocatorType
    long_only: bool = True
    beta_neutral: bool = False
    k_top: PositiveInt | None = Field(default=None)
    temperature: PositiveFloat = Field(default=1.0)
    risk_halflife: PositiveInt = Field(default=12)
    trade_threshold: float = Field(
        default=0.0,
        ge=0.0,
        alias=AliasChoices("trade_threshold", "trade_epsilon"),
    )
    hold_cash_threshold: float = Field(default=0.0, ge=0.0)
    smooth_lambda: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_target_ann: float = Field(default=0.0, ge=0.0)
    weight_cap: float = Field(default=1.0, ge=0.0)
    cost_bps: float = Field(default=10.0, ge=0.0)
    borrow_cap: float | None = Field(default=None, ge=0.0)
    floor_cash: float | None = Field(default=None)
    temperature_grid: list[PositiveFloat] | None = Field(default=None)

    model_config = {"populate_by_name": True}

    @field_validator("k_top")
    @classmethod
    def _validate_k_top(
        cls, value: PositiveInt | None, info: dict[str, Any]
    ) -> PositiveInt | None:
        allocator = info.data.get("allocator") if getattr(info, "data", None) else None
        if allocator == AllocatorType.TOPK and value is None:
            raise ValueError("k_top must be provided when allocator is 'topk'")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, temperature: float, info: dict[str, Any]) -> float:
        allocator = info.data.get("allocator") if getattr(info, "data", None) else None
        if allocator == AllocatorType.SOFTMAX and temperature <= 0:
            raise ValueError("temperature must be > 0 for softmax allocator")
        return temperature


class ConfigModel(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    costs: CostsConfig
    portfolio: PortfolioConfig
    lags: dict[str, NonNegativeInt]

    model_config = {"extra": "allow"}


def load_config(path: Path | str) -> ConfigModel:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping at the root.")
    return ConfigModel.model_validate(payload)


__all__ = [
    "AllocatorType",
    "ConfigModel",
    "CostsConfig",
    "DataConfig",
    "PortfolioConfig",
    "ProjectConfig",
    "SlippageModel",
    "load_config",
]
