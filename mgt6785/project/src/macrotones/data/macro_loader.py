from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

WINDOW_MONTHS: Final[int] = 120
MIN_PERIODS: Final[int] = 24
FRED_SOURCES: Final[tuple[Path, ...]] = (
    Path("data/raw/fred_monthly.parquet"),
    Path("data/processed/panel.parquet"),
)
CPI_CANDIDATES: Final[tuple[str, ...]] = ("CPI", "CPIAUCSL")
GDP_CANDIDATES: Final[tuple[str, ...]] = ("GDPC1", "GDP", "GDPPOT", "INDPRO")
TEN_YEAR_CANDIDATES: Final[tuple[str, ...]] = ("DGS10", "GS10")
TWO_YEAR_CANDIDATES: Final[tuple[str, ...]] = ("DGS2", "GS2", "DGS5", "DGS1")
UNRATE_COLUMNS: Final[tuple[str, ...]] = ("UNRATE", "UNEMPLOY")


def _month_end(date: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    return ts.to_period("M").to_timestamp("M")


def _first_available(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            series = df[col].astype(float)
            return series
    return None


@lru_cache(maxsize=1)
def _load_fred() -> pd.DataFrame:
    for path in FRED_SOURCES:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        monthly = df.resample("ME").last()
        monthly = monthly.ffill()
        lower = monthly.quantile(0.01)
        upper = monthly.quantile(0.99)
        monthly = monthly.clip(lower=lower, upper=upper, axis=1)
        return monthly
    raise FileNotFoundError(
        "Could not locate a FRED source. Expected one of: "
        + ", ".join(str(p) for p in FRED_SOURCES)
    )


def _build_feature_frame() -> pd.DataFrame:
    fred = _load_fred()
    features = pd.DataFrame(index=fred.index)

    cpi_series = _first_available(fred, CPI_CANDIDATES)
    if cpi_series is None:
        raise ValueError("CPI series missing from FRED dataset.")
    features["cpi_yoy"] = cpi_series.pct_change(12) * 100.0

    gdp_series = _first_available(fred, GDP_CANDIDATES)
    if gdp_series is None:
        gdp_series = cpi_series
    gdp_freq = pd.infer_freq(gdp_series.index)
    if gdp_freq and gdp_freq.upper().startswith("Q"):
        gdp_series = (
            gdp_series.to_period("Q").to_timestamp("Q", "end").resample("ME").ffill()
        )
    features["gdp_real"] = gdp_series.pct_change(12) * 100.0

    ten_year = _first_available(fred, TEN_YEAR_CANDIDATES)
    two_year = _first_available(fred, TWO_YEAR_CANDIDATES)
    if ten_year is None:
        raise ValueError("10-year Treasury series missing from FRED dataset.")
    if two_year is None:
        two_year = pd.Series(0.0, index=ten_year.index)
    spread = ten_year.reindex(fred.index).ffill() - two_year.reindex(fred.index).ffill()
    features["y10y_2y_spread"] = spread

    unrate = _first_available(fred, UNRATE_COLUMNS)
    if unrate is None:
        raise ValueError("Unemployment rate series missing from FRED dataset.")
    features["unemp"] = unrate

    return features


@lru_cache(maxsize=1)
def _feature_and_zscores() -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = _build_feature_frame()
    rolling_mean = feats.rolling(WINDOW_MONTHS, min_periods=MIN_PERIODS).mean()
    rolling_std = feats.rolling(WINDOW_MONTHS, min_periods=MIN_PERIODS).std(ddof=0)
    z = (feats - rolling_mean) / rolling_std
    z = z.replace([np.inf, -np.inf], np.nan)
    return feats, z


def _macro_regime_index(zscores: pd.DataFrame, date: pd.Timestamp) -> float:
    ts = _month_end(date)
    window = zscores.loc[:ts].tail(WINDOW_MONTHS).dropna()
    if window.empty:
        return float("nan")
    pca = PCA(n_components=1, random_state=42)
    pca.fit(window.values)
    latest = zscores.loc[:ts].iloc[-1]
    if latest.isna().any():
        latest = window.iloc[-1]
    comp = float(pca.transform(latest.to_numpy(dtype=float).reshape(1, -1))[0, 0])
    if "gdp_real" in window.columns:
        gdp_idx = list(window.columns).index("gdp_real")
        if pca.components_[0, gdp_idx] < 0:
            comp *= -1.0
    return comp


def get_macro_features(date: pd.Timestamp) -> dict[str, float]:
    """
    Return macro features and Macro Regime Index for the requested month-end date.
    """
    feats, zscores = _feature_and_zscores()
    ts = _month_end(pd.Timestamp(date))
    if ts not in feats.index:
        mask = feats.index[feats.index <= ts]
        if mask.empty:
            raise ValueError(f"No macro observations available on/before {ts.date()}.")
        ts = mask[-1]
    row = feats.loc[ts]
    mri = _macro_regime_index(zscores, ts)
    out = {
        "cpi_yoy": float(row["cpi_yoy"]),
        "gdp_real": float(row["gdp_real"]),
        "y10y_2y_spread": float(row["y10y_2y_spread"]),
        "unemp": float(row["unemp"]),
        "mri": float(mri),
    }
    for key, value in out.items():
        if not np.isfinite(value):
            out[key] = float("nan")
    return out
