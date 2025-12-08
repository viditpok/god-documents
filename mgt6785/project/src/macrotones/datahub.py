from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from macrotones.api import loader

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_REGIME_STATE_PATH = CACHE_DIR / "regime_states.parquet"


@lru_cache(maxsize=8)
def load_backtest(config_path: Path | None = None) -> pd.DataFrame:
    path = config_path or loader.DEFAULT_CONFIG
    return loader.load_backtest(path)


@lru_cache(maxsize=8)
def load_macro_panel(config_path: Path | None = None) -> pd.DataFrame:
    path = config_path or loader.DEFAULT_CONFIG
    return loader.load_macro_panel(path)


@lru_cache(maxsize=8)
def load_llm_log(config_path: Path | None = None) -> pd.DataFrame:
    path = config_path or loader.DEFAULT_CONFIG
    return loader.load_llm_policy_log(path)


def load_factor_data(config_path: Path | None = None) -> pd.DataFrame:
    cfg_path = config_path or loader.DEFAULT_CONFIG
    cfg = loader._load_config(cfg_path)
    ff_path = Path(cfg.data.out_raw) / "ff" / "ff_monthly.parquet"
    if not ff_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ff_path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_macro_data(config_path: Path | None = None) -> pd.DataFrame:
    cfg_path = config_path or loader.DEFAULT_CONFIG
    cfg = loader._load_config(cfg_path)
    panel_path = Path(cfg.data.out_processed) / "panel.parquet"
    if not panel_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(panel_path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_tone_signals(config_path: Path | None = None) -> pd.DataFrame:
    cfg_path = config_path or loader.DEFAULT_CONFIG
    cfg = loader._load_config(cfg_path)
    interim = Path(cfg.data.out_interim)
    tone_path = interim / "nlp_regime_scores.parquet"
    if not tone_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(tone_path)
    if "doc_month" in df.columns:
        df["doc_month"] = pd.to_datetime(df["doc_month"])
        df = df.set_index("doc_month")
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def cache_frame(df: pd.DataFrame, name: str) -> Path:
    cache_path = CACHE_DIR / f"{name}.pkl"
    df.to_pickle(cache_path)
    return cache_path


def load_cached_frame(name: str) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / f"{name}.pkl"
    if not cache_path.exists():
        return None
    return pd.read_pickle(cache_path)


def cache_regime_correlation(df: pd.DataFrame) -> Path:
    return cache_frame(df, "regime_corr")


def load_regime_correlation() -> pd.DataFrame | None:
    return load_cached_frame("regime_corr")


def cache_regime_states(df: pd.DataFrame) -> Path:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.to_parquet(_REGIME_STATE_PATH)
    return _REGIME_STATE_PATH


def load_regime_states() -> pd.DataFrame | None:
    if not _REGIME_STATE_PATH.exists():
        return None
    df = pd.read_parquet(_REGIME_STATE_PATH)
    df.index = pd.to_datetime(df.index)
    return df


def cache_joblib(obj: Any, name: str) -> Path:
    cache_path = CACHE_DIR / f"{name}.joblib"
    joblib.dump(obj, cache_path)
    return cache_path


def load_joblib(name: str) -> Any | None:
    cache_path = CACHE_DIR / f"{name}.joblib"
    if not cache_path.exists():
        return None
    return joblib.load(cache_path)


__all__ = [
    "cache_frame",
    "cache_joblib",
    "cache_regime_correlation",
    "cache_regime_states",
    "load_backtest",
    "load_cached_frame",
    "load_factor_data",
    "load_joblib",
    "load_llm_log",
    "load_macro_data",
    "load_macro_panel",
    "load_regime_correlation",
    "load_regime_states",
    "load_tone_signals",
]
