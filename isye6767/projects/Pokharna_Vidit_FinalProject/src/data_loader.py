"""Utilities for loading and serving market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence

import pandas as pd

from .config import ProjectConfig


@dataclass
class MarketDataLoader:
    """Loads price and universe data and exposes handy accessors."""

    config: ProjectConfig

    def __post_init__(self) -> None:
        self.price_df = self._load_price_data(self.config.price_path)
        self.returns_df = self.price_df.pct_change().dropna(how="all")
        self.universe_df = self._load_universe(self.config.universe_path)

    @staticmethod
    def _load_price_data(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
        df = df.drop(columns=["startTime"])
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        df = df.set_index("timestamp")
        # Coerce token columns to numeric and forward-fill sporadic gaps.
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.sort_index()
        df = df.ffill()
        return df

    @staticmethod
    def _load_universe(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
        df = df.drop(columns=["startTime"])
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        df = df.set_index("timestamp")
        return df

    def get_common_index(self) -> pd.DatetimeIndex:
        """Return the intersection of price and universe timestamps."""
        idx = self.price_df.index.intersection(self.universe_df.index)
        return idx.sort_values()

    def iter_time_range(self, start: datetime, end: datetime) -> Iterable[pd.Timestamp]:
        """Yield timestamps between start and end inclusive."""
        idx = self.get_common_index()
        mask = (idx >= start) & (idx <= end)
        return idx[mask]

    def get_universe_tokens(self, timestamp: pd.Timestamp) -> List[str]:
        """Return the list of universe tokens at the supplied timestamp."""
        if timestamp not in self.universe_df.index:
            return []
        row = self.universe_df.loc[timestamp]
        tokens = [str(tok) for tok in row.values if isinstance(tok, str)]
        return tokens

    def get_prices(self, timestamp: pd.Timestamp, tokens: Sequence[str]) -> pd.Series:
        """Return spot prices for the provided tokens."""
        available = [t for t in tokens if t in self.price_df.columns]
        if not available:
            return pd.Series(dtype=float)
        if timestamp not in self.price_df.index:
            return pd.Series(dtype=float)
        return self.price_df.loc[timestamp, available]

    def get_price_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tokens: Sequence[str],
    ) -> pd.DataFrame:
        """Slice prices between start and end inclusive."""
        available = [t for t in tokens if t in self.price_df.columns]
        if not available:
            return pd.DataFrame()
        return self.price_df.loc[start:end, available]

    def get_return_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tokens: Sequence[str],
    ) -> pd.DataFrame:
        """Slice returns between start and end inclusive."""
        available = [t for t in tokens if t in self.returns_df.columns]
        if not available:
            return pd.DataFrame()
        df = self.returns_df.loc[start:end, available]
        return df
