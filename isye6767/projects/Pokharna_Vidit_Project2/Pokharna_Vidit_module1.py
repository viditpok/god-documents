from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# backtrader feed class for reading csv data with multiple schema support
class CustomCSVData(bt.feeds.GenericCSVData):

    params = (
        ("csv_type", "yahoo"),
        ("nullvalue", 0.0),
        ("dtformat", ("%Y-%m-%d")),
        ("datetime", 0),
        ("timeframe", bt.TimeFrame.Days),
        ("compression", 1),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", -1),
    )

    _csv_type_params = {
        "yahoo": dict(datetime=0, open=3, high=1, low=2, close=4, volume=5, openinterest=-1),
        "tushare": dict(datetime=0, open=1, high=3, low=4, close=2, volume=5, openinterest=-1),
        "ercot": dict(datetime=0, time=-1, open=2, high=2, low=2, close=2, volume=-1, openinterest=-1),
    }

    def __init__(self, *args, **kwargs):
        csv_type = kwargs.pop("csv_type", self.p.csv_type)
        params = self._csv_type_params.get(
            csv_type, self._csv_type_params["yahoo"])
        for key, value in params.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)


# configuration for market data fetching and caching
@dataclass
class MarketDataConfig:

    start_date: str
    end_date: str
    data_directories: Iterable[Path] = field(
        default_factory=lambda: (Path("data/stock_dfs"), Path("data"))
    )
    cache_directory: Path = Path("cache")
    min_history: int = 250


# data loading: handles downloading, caching, and normalizing historical price data
class MarketDataManager:

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.config.cache_directory.mkdir(parents=True, exist_ok=True)

    def _candidate_files(self, ticker: str) -> List[Path]:
        candidates = []
        for directory in self.config.data_directories:
            path = directory / f"{ticker}.csv"
            if path.exists():
                candidates.append(path)
        disk_cache = self.config.cache_directory / f"{ticker}.csv"
        if disk_cache.exists():
            candidates.insert(0, disk_cache)
        return candidates

    # preprocessing: normalize column names to standard format
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [str(levels[0]) for levels in df.columns]
        rename_map = {
            "Date": "date",
            "date": "date",
            "Datetime": "date",
            "Open": "open",
            "open": "open",
            "High": "high",
            "high": "high",
            "Low": "low",
            "low": "low",
            "Close": "close",
            "close": "close",
            "Adj Close": "adj_close",
            "AdjClose": "adj_close",
            "adj_close": "adj_close",
            "Volume": "volume",
            "volume": "volume",
        }
        df = df.rename(columns=rename_map)
        if "adj_close" not in df and "close" in df:
            df["adj_close"] = df["close"]
        required_columns = ["date", "open", "high",
                            "low", "close", "adj_close", "volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing} in supplied dataframe.")
        # convert date to datetime and numeric columns to float
        df["date"] = pd.to_datetime(df["date"])
        for column in ["open", "high", "low", "close", "adj_close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.sort_values("date")
        return df[required_columns]

    def _download_from_yahoo(self, ticker: str) -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            raise RuntimeError(f"Unable to download data for ticker {ticker}")
        data = data.reset_index()

        def _normalize_name(col) -> str:
            base = col[0] if isinstance(col, tuple) and col else col
            return str(base).lower().replace(" ", "_")

        standard_names = {col: _normalize_name(col) for col in data.columns}
        data = data.rename(columns=standard_names)
        return self._normalize_columns(data)

    def load(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        if not force_download:
            for path in self._candidate_files(ticker):
                df = pd.read_csv(path)
                df = self._normalize_columns(df)
                if len(df) >= self.config.min_history:
                    return df
        try:
            df = self._download_from_yahoo(ticker)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download data for ticker {ticker}: {exc}") from None
        output_path = self.config.cache_directory / f"{ticker}.csv"
        df.to_csv(output_path, index=False)
        return df

    def export_metadata(self, tickers: Iterable[str], output_path: Path) -> None:
        summary: Dict[str, Dict[str, Optional[str]]] = {}
        for ticker in tickers:
            try:
                df = self.load(ticker)
            except Exception as exc:
                summary[ticker] = {"error": str(
                    exc), "start": None, "end": None}
                continue
            summary[ticker] = {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
                "rows": len(df),
            }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


# preprocessing: handles data cleaning, filling missing values, and feature scaling
class DataPreprocessor:

    def __init__(self, fill_method: str = "ffill", scaler: str = "standard"):
        self.fill_method = fill_method
        self.scaler_choice = scaler
        self.scalers: Dict[str, StandardScaler] = {}

    def _get_scaler(self) -> StandardScaler:
        if self.scaler_choice == "standard":
            return StandardScaler()
        if self.scaler_choice == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unsupported scaler: {self.scaler_choice}")

    # preprocessing: clean price data by removing duplicates, setting business day frequency, and filling missing values
    def clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates("date").sort_values("date")
        df = df.set_index("date").asfreq("B")
        # fill missing values based on specified method
        if self.fill_method == "ffill":
            df = df.ffill().bfill()
        elif self.fill_method == "bfill":
            df = df.bfill().ffill()
        else:
            df = df.interpolate().ffill().bfill()
        return df

    def split_train_test(self, df: pd.DataFrame, train_ratio: float = 0.6):
        boundary = int(len(df) * train_ratio)
        train = df.iloc[:boundary]
        test = df.iloc[boundary:]
        return train, test

    # preprocessing: split dataset into train/validation/test sets (60/20/20 by default)
    def split_datasets(
        self, df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2
    ):
        total = len(df)
        if total < 3:
            raise ValueError("Not enough samples to perform 60/20/20 split.")
        train_end = max(1, int(total * train_ratio))
        val_end = train_end + max(1, int(total * val_ratio))
        if val_end >= total:
            val_end = total - 1
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        if len(test) == 0:
            test = df.iloc[-1:]
            val = df.iloc[train_end:-1]
        return train, val, test

    # preprocessing: scale features using standard scaler or minmax scaler
    def scale(self, train_features: pd.DataFrame, test_features: pd.DataFrame):
        scaler = self._get_scaler()
        scaled_train = scaler.fit_transform(train_features.values)
        scaled_test = scaler.transform(test_features.values)
        return (
            pd.DataFrame(scaled_train, index=train_features.index,
                         columns=train_features.columns),
            pd.DataFrame(scaled_test, index=test_features.index,
                         columns=test_features.columns),
            scaler,
        )

    # preprocessing: scale train, validation, and test features using same scaler fit on training data
    def scale_three(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ):
        scaler = self._get_scaler()
        scaled_train = scaler.fit_transform(train_features.values)
        scaled_val = scaler.transform(val_features.values)
        scaled_test = scaler.transform(test_features.values)
        return (
            pd.DataFrame(scaled_train, index=train_features.index,
                         columns=train_features.columns),
            pd.DataFrame(scaled_val, index=val_features.index,
                         columns=val_features.columns),
            pd.DataFrame(scaled_test, index=test_features.index,
                         columns=test_features.columns),
            scaler,
        )


# feature creation: generates technical indicators and price-based features for model training
class PriceFeatureEngineer:

    def __init__(self, windows: Iterable[int] = (5, 10, 21, 63)):
        self.windows = sorted(set(windows))

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # feature creation: build feature frame with technical indicators and target variable
    def build_feature_frame(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        features = pd.DataFrame(index=df.index)
        # basic return features
        features["return_1d"] = df["close"].pct_change()
        features["log_return_5d"] = np.log(df["close"] / df["close"].shift(5))
        features["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / df[
            "volume"
        ].rolling(20).std()
        # rolling window features: sma, ema, momentum, volatility
        for window in self.windows:
            features[f"sma_{window}"] = df["close"].rolling(window).mean()
            features[f"ema_{window}"] = df["close"].ewm(
                span=window, adjust=False).mean()
            features[f"momentum_{window}"] = df["close"].pct_change(window)
            features[f"volatility_{window}"] = df["close"].pct_change().rolling(
                window).std()
        # technical indicators: rsi, macd, bollinger bands
        features["rsi_14"] = self._rsi(df["close"])
        macd_fast = df["close"].ewm(span=12, adjust=False).mean()
        macd_slow = df["close"].ewm(span=26, adjust=False).mean()
        features["macd"] = macd_fast - macd_slow
        features["macd_signal"] = features["macd"].ewm(
            span=9, adjust=False).mean()
        features["bollinger_high"] = df["close"].rolling(
            20).mean() + 2 * df["close"].rolling(20).std()
        features["bollinger_low"] = df["close"].rolling(
            20).mean() - 2 * df["close"].rolling(20).std()
        # combine price data with features and create target (next day price up/down)
        dataset = pd.concat(
            [df[["open", "high", "low", "close", "adj_close", "volume"]], features], axis=1)
        dataset["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        dataset = dataset.dropna()
        return dataset


def load_tickers(file_path: Path) -> List[str]:
    df = pd.read_csv(file_path)
    tickers = [
        str(t).strip().upper()
        for t in df.iloc[:, 0].tolist()
        if isinstance(t, str) and t.strip()
    ]
    return tickers


__all__ = [
    "CustomCSVData",
    "MarketDataConfig",
    "MarketDataManager",
    "DataPreprocessor",
    "PriceFeatureEngineer",
    "load_tickers",
]
