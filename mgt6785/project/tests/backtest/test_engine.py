from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from macrotones.backtest import engine
from macrotones.backtest.config import BacktestConfig

np.seterr(invalid="ignore")


def _write_config(base_dir: str, cfg: dict) -> None:
    cfg_path = Path(base_dir) / "project.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream)


def _prepare_env(tmp_path) -> dict[str, str]:
    config_dir = tmp_path / "config"
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    ff_dir = raw_dir / "ff"
    for path in (config_dir, processed_dir, raw_dir, interim_dir, ff_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "config": str(config_dir),
        "processed": str(processed_dir),
        "raw": str(raw_dir),
        "interim": str(interim_dir),
        "ff": str(ff_dir),
    }


def test_engine_main_turnover_and_cost(monkeypatch, tmp_path) -> None:
    paths = _prepare_env(tmp_path)
    cfg = {
        "project": {
            "name": "Test",
            "start": "2020-01-01",
            "end": "2020-12-31",
            "rebalance": "M",
            "rf_col": "RF",
        },
        "data": {
            "fred_series": ["CPI"],
            "out_processed": paths["processed"],
            "out_raw": paths["raw"],
            "out_interim": paths["interim"],
        },
        "model": {
            "horizon_months": 1,
            "cv_splits": 2,
            "min_train_months": 1,
            "ensemble": ["ridge"],
            "ridge_alpha": 3.0,
            "ridge_alpha_grid": [3.0],
        },
        "portfolio": {
            "allocator": "softmax",
            "long_only": True,
            "temperature": 0.5,
            "temperature_grid": [0.5],
            "trade_threshold": 0.0,
        },
        "costs": {"model": "fixed", "fixed_bps": 50},
        "lags": {"fred_months": 1},
    }
    config_path = Path(paths["config"]) / "project.yaml"
    _write_config(paths["config"], cfg)

    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    ff = pd.DataFrame(
        {
            "HML": [0.02, 0.03],
            "SMB": [-0.01, 0.02],
            "Mkt_RF": [0.03, 0.02],
            "UMD": [0.01, 0.01],
            "RF": [0.001, 0.001],
        },
        index=idx,
    )
    ff.to_parquet(f"{paths['ff']}/ff_monthly.parquet")

    preds_df = pd.DataFrame(
        {
            "HML": [0.05, 0.04],
            "SMB": [0.02, 0.01],
            "UMD": [0.01, 0.02],
            "Mkt_RF": [0.03, 0.02],
            "RF": [0.001, 0.001],
        },
        index=idx,
    )

    weights = [
        pd.Series({"HML": 0.5, "SMB": 0.5}, name=idx[0]),
        pd.Series({"HML": 0.2, "SMB": 0.8}, name=idx[1]),
    ]
    call_index = {"i": 0}

    def fake_build_weights(mu, hist, cfg_port, prev_w=None, risk_cov=None):
        current = weights[call_index["i"] % len(weights)]
        prev = (
            prev_w.reindex(current.index).fillna(0.0)
            if prev_w is not None
            else pd.Series(0.0, index=current.index)
        )
        turnover = float((current - prev).abs().sum())
        score = mu
        call_index["i"] += 1
        return current, turnover, score

    def fake_train_models(
        ridge_alpha=None,
        capture_coeffs=False,
        save_outputs=True,
        config_path: Path | None = None,
    ):
        if save_outputs:
            Path(paths["processed"]).mkdir(parents=True, exist_ok=True)
            preds_df.to_parquet(Path(paths["processed"]) / "preds.parquet")
        return preds_df.copy(), {}

    monkeypatch.setattr(engine, "build_weights", fake_build_weights)
    monkeypatch.setattr(engine, "train_models", fake_train_models)
    monkeypatch.setattr(engine, "plot_ridge_coeffs", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)

    engine.main(config_path=config_path)

    bt = pd.read_parquet(f"{paths['processed']}/backtest.parquet")
    expected_turnover = [1.0, 0.6]
    np.testing.assert_allclose(bt["turnover"].values, expected_turnover)
    slippage_expected = np.array(expected_turnover) * cfg["costs"]["fixed_bps"] * 1e-4
    np.testing.assert_allclose(bt["slippage"].values, slippage_expected)
    portfolio_cost_bps = float(cfg["portfolio"].get("cost_bps", 10.0))
    transaction_expected = slippage_expected + np.array(expected_turnover) * (
        portfolio_cost_bps * 1e-4
    )
    np.testing.assert_allclose(bt["transaction_cost"].values, transaction_expected)
    rf_monthly = ff["RF"].values / 1200.0
    expected_gross = rf_monthly + np.array(
        [
            0.5 * ff.loc[idx[0], "HML"] + 0.5 * ff.loc[idx[0], "SMB"],
            0.2 * ff.loc[idx[1], "HML"] + 0.8 * ff.loc[idx[1], "SMB"],
        ]
    )
    np.testing.assert_allclose(bt["rf"].values, rf_monthly, rtol=1e-12)
    np.testing.assert_allclose(bt["gross_ret"].values, expected_gross, rtol=1e-9)
    expected_net = expected_gross - transaction_expected
    np.testing.assert_allclose(bt["net_ret"].values, expected_net)
    np.testing.assert_allclose(bt["net_excess"].values, expected_net - rf_monthly)
    assert bt["trade_mask"].tolist() == [True, True]


def test_allocator_behaviour_and_temperature() -> None:
    scores = pd.Series({"A": 0.04, "B": 0.02, "C": 0.01})
    cfg_low_temp = {"allocator": "softmax", "temperature": 0.5, "long_only": True}
    cfg_high_temp = {"allocator": "softmax", "temperature": 5.0, "long_only": True}

    hist_df = pd.DataFrame(
        {
            "A": np.linspace(-0.02, 0.03, 20),
            "B": np.linspace(0.01, -0.015, 20),
            "C": np.linspace(0.005, -0.005, 20),
        }
    )

    w_low, _, _ = engine.build_weights(scores, hist_df, cfg_low_temp)
    w_high, _, _ = engine.build_weights(scores, hist_df, cfg_high_temp)

    spread_low = float(w_low.max() - w_low.min())
    spread_high = float(w_high.max() - w_high.min())
    assert spread_low > spread_high
    assert w_low.max() > 0

    cfg_topk = {"allocator": "topk", "k_top": 1, "long_only": True}
    w_topk, _, _ = engine.build_weights(scores, hist_df, cfg_topk)
    positive = w_topk[w_topk > 0]
    assert len(positive) <= 1
    if not positive.empty:
        vol = hist_df.rolling(12).std().iloc[-1].reindex(scores.index)
        vol = vol.replace(0, np.nan)
        expected = (scores / vol).fillna(0.0).idxmax()
        assert positive.index[0] == expected


def test_risk_targeting_and_cap() -> None:
    scores = pd.Series({"A": 0.05, "B": 0.04, "C": 0.03})
    hist_returns = pd.DataFrame(
        {
            "A": [0.1, 0.2, -0.15, 0.05],
            "B": [0.3, -0.25, 0.4, -0.2],
            "C": [0.05, 0.02, -0.01, 0.03],
        }
    )

    cfg_loose = {
        "allocator": "softmax",
        "temperature": 1.0,
        "long_only": False,
        "risk_target_ann": 1.0,
        "weight_cap": 1.0,
    }
    cfg_strict = {
        "allocator": "softmax",
        "temperature": 1.0,
        "long_only": False,
        "risk_target_ann": 0.1,
        "weight_cap": 0.4,
    }

    w_loose, _, _ = engine.build_weights(scores, hist_returns, cfg_loose)
    w_strict, _, _ = engine.build_weights(scores, hist_returns, cfg_strict)

    assert w_strict.abs().sum() <= w_loose.abs().sum()
    assert w_strict.le(0.4 + 1e-9).all()


def test_seed_determinism(monkeypatch, tmp_path) -> None:
    paths = _prepare_env(tmp_path)
    cfg = {
        "project": {
            "name": "Test",
            "start": "2021-01-01",
            "end": "2021-12-31",
            "rebalance": "M",
            "rf_col": "RF",
        },
        "data": {
            "fred_series": ["CPI"],
            "out_processed": paths["processed"],
            "out_raw": paths["raw"],
            "out_interim": paths["interim"],
        },
        "model": {
            "horizon_months": 1,
            "cv_splits": 2,
            "min_train_months": 1,
            "ensemble": ["ridge"],
            "ridge_alpha": 3.0,
            "ridge_alpha_grid": [3.0],
        },
        "portfolio": {
            "allocator": "softmax",
            "long_only": True,
            "temperature": 0.5,
            "temperature_grid": [0.5],
            "trade_threshold": 0.0,
        },
        "costs": {"model": "fixed", "fixed_bps": 0},
        "lags": {"fred_months": 1},
    }
    config_path = Path(paths["config"]) / "project.yaml"
    _write_config(paths["config"], cfg)

    idx = pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"])
    ff = pd.DataFrame(
        {
            "HML": [0.01, -0.02, 0.03],
            "SMB": [0.02, 0.01, -0.01],
            "UMD": [0.0, 0.01, -0.02],
            "Mkt_RF": [0.02, 0.02, 0.02],
            "RF": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    ff.to_parquet(f"{paths['ff']}/ff_monthly.parquet")

    preds_df = pd.DataFrame(
        {
            "HML": [0.03, 0.02, 0.01],
            "SMB": [0.01, 0.0, -0.01],
            "UMD": [-0.005, 0.0, 0.005],
            "Mkt_RF": [0.02, 0.02, 0.02],
            "RF": [0.0, 0.0, 0.0],
        },
        index=idx,
    )

    def fake_train_models(
        ridge_alpha=None,
        capture_coeffs=False,
        save_outputs=True,
        config_path: Path | None = None,
    ):
        if save_outputs:
            Path(paths["processed"]).mkdir(parents=True, exist_ok=True)
            preds_df.to_parquet(Path(paths["processed"]) / "preds.parquet")
        return preds_df.copy(), {}

    monkeypatch.setattr(engine, "train_models", fake_train_models)
    monkeypatch.setattr(engine, "plot_ridge_coeffs", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)

    result_frames = []
    for _ in range(2):
        backtest_path = Path(paths["processed"]) / "backtest.parquet"
        if backtest_path.exists():
            backtest_path.unlink()
        engine.main(config_path=config_path)
        result_frames.append(pd.read_parquet(backtest_path))

    pd.testing.assert_frame_equal(result_frames[0], result_frames[1])


def test_beta_neutral_reduces_market_beta() -> None:
    dates = pd.date_range("2015-01-31", periods=48, freq="ME")
    market = np.linspace(-0.02, 0.03, len(dates)) + np.random.default_rng(0).normal(
        0, 0.01, len(dates)
    )
    hml = np.sin(np.arange(len(dates)) / 3) * 0.015
    ff = pd.DataFrame(
        {
            "Mkt_RF": market,
            "HML": hml,
            "SMB": np.zeros(len(dates)),
            "UMD": np.zeros(len(dates)),
            "RF": np.full(len(dates), 1.2),
        },
        index=dates,
    )
    preds = pd.DataFrame(
        {
            "Mkt_RF": np.full(len(dates), 0.03),
            "HML": np.full(len(dates), 0.01),
            "SMB": np.zeros(len(dates)),
            "UMD": np.zeros(len(dates)),
            "RF": np.full(len(dates), 1.2),
        },
        index=dates,
    )
    cfg_port = {
        "allocator": "sharpe",
        "long_only": False,
        "trade_threshold": 0.0,
        "beta_neutral": True,
    }
    cfg_costs = {"model": "fixed", "fixed_bps": 0.0}

    cfg_base = BacktestConfig(
        rebalance_freq="M",
        trade_eps=0.0,
        rf_col="RF",
        beta_neutral=False,
    )
    cfg_neutral = BacktestConfig(
        rebalance_freq="M",
        trade_eps=0.0,
        rf_col="RF",
        beta_neutral=True,
    )

    bt_base = engine.run_backtest(
        preds,
        ff,
        ["Mkt_RF", "SMB", "HML", "UMD"],
        cfg_port,
        cfg_costs,
        cfg_base,
    )
    bt_neutral = engine.run_backtest(
        preds,
        ff,
        ["Mkt_RF", "SMB", "HML", "UMD"],
        cfg_port,
        cfg_costs,
        cfg_neutral,
    )

    def _beta(series: pd.Series, market_series: pd.Series) -> float:
        y = series.values
        X = np.column_stack([market_series.values, np.ones_like(y)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coef[0])

    market_series = ff["Mkt_RF"].reindex(bt_base.index)
    beta_base = _beta(bt_base["net_ret"], market_series)
    beta_neutral = _beta(bt_neutral["net_ret"], market_series)

    assert abs(beta_base) > 0.3
    assert abs(beta_neutral) < 0.1
