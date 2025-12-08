from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd

from macrotones.config.schema import ConfigModel, load_config

DEFAULT_CONFIG = Path("config/project.yaml")


def list_configurations(config_root: Path | str = "config") -> dict[str, Path]:
    root = Path(config_root)
    configs = sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))
    return {cfg.stem: cfg for cfg in configs}


@lru_cache(maxsize=32)
def _load_config(path: Path | str) -> ConfigModel:
    return load_config(path)


def processed_path(config_path: Path | str = DEFAULT_CONFIG) -> Path:
    cfg = _load_config(Path(config_path))
    path = Path(cfg.data.out_processed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_parquet(name: str, config_path: Path | str) -> pd.DataFrame:
    path = processed_path(config_path) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_csv(
    name: str,
    config_path: Path | str,
    index_col: str | None = None,
) -> pd.DataFrame:
    path = processed_path(config_path) / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def load_backtest(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_parquet("backtest.parquet", config_path)


def load_ic_summary(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_csv("ic_summary.csv", config_path, index_col="factor")


def load_ic_rolling(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_parquet("ic_rolling.parquet", config_path)


def load_ablation(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_csv("ablation_summary.csv", config_path)


def load_sweep(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_csv("sweep_summary.csv", config_path)


def load_turnover(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    return _load_parquet("turnover_rolling.parquet", config_path)


def load_alpha_decomposition(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return _load_parquet("alpha_decomposition.parquet", config_path)


def load_tracking_error(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return _load_parquet("tracking_error.parquet", config_path)


def load_excess_market(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return _load_parquet("excess_market.parquet", config_path)


def load_drawdown_attribution(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return _load_csv("drawdown_attribution.csv", config_path)


def load_bootstrap_sharpe(
    config_path: Path | str = DEFAULT_CONFIG,
) -> dict[str, float]:
    path = processed_path(config_path) / "bootstrap_sharpe.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_regime_information(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return _load_csv("regime_information.csv", config_path, index_col="regime")


def load_regime_summary(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    path = processed_path(config_path) / "regime_summary.csv"
    return _cached_regime_summary(str(path), _stamp(path))


def load_regime_coverage(config_path: Path | str = DEFAULT_CONFIG) -> pd.DataFrame:
    path = processed_path(config_path) / "regime_coverage.csv"
    return _cached_regime_coverage(str(path), _stamp(path))


def load_image(path: Path) -> Path | None:
    return path if path.exists() else None


def get_report_images(
    config_path: Path | str = DEFAULT_CONFIG,
) -> dict[str, Path | None]:
    base = processed_path(config_path)
    return {
        "tearsheet": load_image(base / "tearsheet.png"),
        "equity": load_image(base / "equity_curve.png"),
        "rolling": load_image(base / "rolling_sharpe.png"),
        "ridge": load_image(base / "ridge_coeffs.png"),
        "regime": load_image(base / "regime_boxplot.png"),
    }


def get_report_assets(
    config_path: Path | str = DEFAULT_CONFIG,
) -> dict[str, Path | None]:
    base = processed_path(config_path)
    tearsheet = load_image(base / "tearsheet.png")
    report_html = base / "report.html"
    if not report_html.exists():
        report_html = None
    return {
        "tearsheet": tearsheet,
        "report_html": report_html,
    }


def pick_series(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default_value: float = 0.0,
) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default_value, index=df.index)


def get_return_series(
    bt: pd.DataFrame, cfg: ConfigModel
) -> tuple[pd.Series, pd.Series]:
    net = pick_series(bt, ["net_ret", "port_ret", "strategy"], default_value=0.0)
    rf = pick_series(bt, [cfg.project.rf_col, "rf"], default_value=0.0)
    return net, rf


def load_market_series(
    index: pd.Index,
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.Series:
    base = processed_path(config_path)
    backtest_path = base / "backtest.parquet"
    if backtest_path.exists():
        bt = pd.read_parquet(backtest_path)
        for col in ["benchmark_ret", "Mkt_RF", "market", "mkt_rf", "market_excess"]:
            if col in bt.columns:
                return bt[col].reindex(index)
    cfg = _load_config(Path(config_path))
    ff_path = Path(cfg.data.out_raw) / "ff" / "ff_monthly.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path)
        ff.index = pd.to_datetime(ff.index)
        return ff.get("Mkt_RF", pd.Series(index=index, dtype=float)).reindex(index)
    return pd.Series(0.0, index=index)


def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=32)
def _cached_regime_summary(
    path: str, _ts: float | None = None
) -> pd.DataFrame:  # pragma: no cover - thin wrapper
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except ValueError:
        df = pd.read_csv(csv_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    if "regime" in df.columns:
        df = df.set_index("regime")
    return df


@lru_cache(maxsize=32)
def _cached_regime_coverage(
    path: str, _ts: float | None = None
) -> pd.DataFrame:  # pragma: no cover - thin wrapper
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    date_cols = [col for col in ("date", "month") if col in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    return df


def load_latest_regime(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_regime_summary(config_path)


def load_latest_regime_coverage(
    config_path: Path | str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_regime_coverage(config_path)
