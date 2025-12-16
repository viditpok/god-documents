from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from macrotones.backtest.config import BacktestConfig
from macrotones.backtest.engine import normalize_costs, run_backtest
from macrotones.backtest.metrics import (
    annualize_monthly,
    max_drawdown,
    sharpe_excess,
)
from macrotones.models.baselines import fit_ridge
from macrotones.models.train import TARGETS, crossval_select
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed

MIN_TRAIN_MONTHS = 36
VALIDATION_MIN_MONTHS = 6


def _load_cfg() -> dict[str, Any]:
    with Path("config/project.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _prepare_panel(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pro = Path(cfg["data"]["out_processed"])
    panel = pd.read_parquet(pro / "panel.parquet").sort_index()
    X = panel.drop(columns=[*TARGETS, "Mkt_RF", "RF"])
    Y = panel[TARGETS]
    return X, Y


def _train_ridge_models(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    alpha: float,
) -> tuple[dict[str, Any], list[str]]:
    usable_cols = [col for col in X_train.columns if X_train[col].notna().any()]
    if not usable_cols:
        raise ValueError("No usable features for ridge training.")
    X_trim = X_train[usable_cols]
    models: dict[str, Any] = {}
    for target in TARGETS:
        y_target = Y_train[target].loc[X_trim.index]
        models[target] = fit_ridge(X_trim, y_target, alpha=alpha)
    return models, usable_cols


def _predict_targets(
    models: dict[str, Any],
    feature_cols: list[str],
    X_eval: pd.DataFrame,
) -> pd.DataFrame:
    X_slice = X_eval.reindex(columns=feature_cols)
    preds = {}
    for target, mdl in models.items():
        preds[target] = mdl.predict(X_slice)
    return pd.DataFrame(preds, index=X_slice.index)


def _build_preds_frame(
    preds: pd.DataFrame,
    ff: pd.DataFrame,
    rf_col: str,
) -> pd.DataFrame:
    aligned = preds.copy()
    aligned["Mkt_RF"] = ff["Mkt_RF"].reindex(aligned.index)
    aligned[rf_col] = ff[rf_col].reindex(aligned.index)
    return aligned.dropna(subset=[rf_col, "Mkt_RF"])


def _adjust_range(
    start: pd.Timestamp,
    offset_months: int,
) -> pd.Timestamp:
    if offset_months == 0:
        return start
    return start + pd.DateOffset(months=offset_months)


def _evaluate_temperature(
    preds: pd.DataFrame,
    ff: pd.DataFrame,
    cfg_port: dict[str, Any],
    cfg_costs: dict[str, Any],
    temp_grid: list[float],
    bt_cfg: BacktestConfig,
) -> float:
    if preds.empty:
        return temp_grid[0]
    factors = [
        col
        for col in ["Mkt_RF", "SMB", "HML", "UMD"]
        if col in preds.columns and col in ff.columns
    ]
    if not factors:
        return temp_grid[0]
    ff_slice = ff.loc[preds.index]

    best_temp = temp_grid[0]
    best_score = -np.inf
    for temp in temp_grid:
        port_cfg = dict(cfg_port)
        port_cfg["temperature"] = temp
        bt = run_backtest(
            preds,
            ff_slice,
            factors,
            port_cfg,
            cfg_costs,
            bt_cfg,
        )
        score = sharpe_excess(bt["net_excess"])
        if score > best_score:
            best_score = score
            best_temp = temp
    return best_temp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", type=int, default=10)
    parser.add_argument("--test_years", type=int, default=3)
    parser.add_argument("--step_years", type=int, default=3)
    parser.add_argument("--start", type=str, default="2000-01-31")
    parser.add_argument("--end", type=str, default="2025-07-31")
    parser.add_argument("--purge", type=int, default=1)
    parser.add_argument("--embargo", type=int, default=1)
    args = parser.parse_args()

    set_global_seed(DEFAULT_SEED)
    cfg = _load_cfg()
    pro = Path(cfg["data"]["out_processed"])
    raw = Path(cfg["data"]["out_raw"])

    X, Y = _prepare_panel(cfg)
    ff = pd.read_parquet(raw / "ff" / "ff_monthly.parquet").sort_index()
    ff.index = pd.to_datetime(ff.index)

    alpha_grid = cfg["model"].get("ridge_alpha_grid") or [
        cfg["model"].get("ridge_alpha", 3.0)
    ]
    temp_grid = cfg["portfolio"].get("temperature_grid") or [
        cfg["portfolio"].get("temperature", 0.5)
    ]
    port_cfg = cfg["portfolio"].copy()
    cost_cfg = normalize_costs(cfg.get("costs", {}))
    rf_col = cfg["project"]["rf_col"]
    beta_neutral = bool(port_cfg.get("beta_neutral", False))
    risk_halflife = int(port_cfg.get("risk_halflife", 12))
    bt_cfg = BacktestConfig(
        rebalance_freq=cfg["project"].get("rebalance", "M"),
        trade_eps=float(port_cfg.get("trade_threshold", 0.0)),
        borrow_cap=port_cfg.get("borrow_cap"),
        floor_cash=port_cfg.get("floor_cash"),
        rf_col=rf_col,
        beta_neutral=beta_neutral,
        risk_halflife=risk_halflife,
        cost_bps=float(port_cfg.get("cost_bps", 10.0)),
        benchmark=cfg["project"].get("benchmark", "SPY"),
    )

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    current = start
    window_id = 0
    rows: list[dict[str, Any]] = []

    while current < end:
        train_end = current + pd.DateOffset(years=args.train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=args.test_years)
        if test_end > end:
            break

        train_cutoff = train_end - pd.DateOffset(months=args.embargo)
        if train_cutoff <= current:
            current += pd.DateOffset(years=args.step_years)
            continue

        X_train = X.loc[current:train_cutoff]
        Y_train = Y.loc[X_train.index]
        if len(X_train) < MIN_TRAIN_MONTHS:
            current += pd.DateOffset(years=args.step_years)
            continue

        best_alpha = crossval_select(
            X_train,
            Y_train,
            alpha_grid,
            DEFAULT_SEED,
            args.purge,
            args.embargo,
        )

        val_window = max(VALIDATION_MIN_MONTHS, args.purge + args.embargo + 1)
        val_start = train_cutoff - pd.DateOffset(months=val_window - 1)
        if val_start <= current:
            val_start = current
        X_val = X_train.loc[val_start:train_cutoff]
        inner_end = val_start - pd.DateOffset(months=args.purge)
        if inner_end <= current:
            inner_end = current
        X_inner = X_train.loc[current:inner_end]
        if X_inner.empty or X_val.empty:
            current += pd.DateOffset(years=args.step_years)
            continue

        models_inner, feature_cols = _train_ridge_models(
            X_inner,
            Y.loc[X_inner.index],
            best_alpha,
        )
        preds_val_targets = _predict_targets(models_inner, feature_cols, X_val)
        preds_val = _build_preds_frame(preds_val_targets, ff, rf_col)
        best_temp = _evaluate_temperature(
            preds_val,
            ff,
            port_cfg,
            cost_cfg,
            temp_grid,
            bt_cfg,
        )

        models_full, feature_cols_full = _train_ridge_models(
            X_train,
            Y_train,
            best_alpha,
        )
        test_start_adj = _adjust_range(test_start, args.purge)
        X_test = X.loc[test_start_adj:test_end]
        if X_test.empty:
            current += pd.DateOffset(years=args.step_years)
            continue
        preds_test_targets = _predict_targets(models_full, feature_cols_full, X_test)
        preds_test = _build_preds_frame(preds_test_targets, ff, rf_col)
        ff_slice = ff.loc[preds_test.index]

        factors = [
            col
            for col in ["Mkt_RF", "SMB", "HML", "UMD"]
            if col in preds_test.columns and col in ff_slice.columns
        ]
        if not factors:
            current += pd.DateOffset(years=args.step_years)
            continue

        port_cfg_eval = dict(port_cfg)
        port_cfg_eval["temperature"] = best_temp

        bt = run_backtest(
            preds_test,
            ff_slice,
            factors,
            port_cfg_eval,
            cost_cfg,
            bt_cfg,
        )

        ann_ret, ann_vol = annualize_monthly(bt["net_ret"])
        sharpe = sharpe_excess(bt["net_excess"])
        max_dd = max_drawdown(bt["net_ret"])

        rows.append(
            {
                "window_id": window_id,
                "train_start": current.date(),
                "train_end": train_cutoff.date(),
                "test_start": test_start_adj.date(),
                "test_end": test_end.date(),
                "inner_best_alpha": best_alpha,
                "inner_best_temperature": best_temp,
                "oos_sharpe": sharpe,
                "oos_ret": ann_ret,
                "oos_vol": ann_vol,
                "maxdd": max_dd,
            }
        )

        window_id += 1
        current += pd.DateOffset(years=args.step_years)

    result_df = pd.DataFrame(rows)
    out_path = pro / "wf_nested.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Nested walk-forward windows: {len(result_df)}")
    print(f"Saved nested WF summary -> {out_path}")


if __name__ == "__main__":
    main()
