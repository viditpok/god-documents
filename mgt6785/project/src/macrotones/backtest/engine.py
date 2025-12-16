from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macrotones.backtest.config import BacktestConfig
from macrotones.backtest.errors import BacktestError
from macrotones.backtest.metrics import annualize_monthly, max_drawdown, sharpe_excess
from macrotones.backtest.performance import generate_performance_artifacts
from macrotones.backtest.utils import (
    apply_trade_threshold,
    compute_slippage,
    enforce_leverage_constraints,
    is_rebalance_date,
    load_benchmark_series,
)
from macrotones.config.schema import ConfigModel, SlippageModel, load_config
from macrotones.core.weights import blend_with_llm
from macrotones.models.train import plot_ridge_coeffs, train_models
from macrotones.portfolio.allocators import build_weights
from macrotones.reports.logs import append_policy_log
from macrotones.risk.model import risk_model
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed

DEFAULT_CONFIG_PATH = Path("config/project.yaml")


def _project_beta_neutral(
    weights: pd.Series,
    cov: pd.DataFrame | None,
    market_col: str | None,
) -> pd.Series:
    if market_col is None or cov is None or market_col not in cov.columns:
        return weights
    aligned = cov.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    if market_col not in aligned.columns:
        return weights
    var_market = float(aligned.loc[market_col, market_col])
    if var_market <= 1e-8 or not np.isfinite(var_market):
        return weights
    beta_vec = aligned[market_col] / var_market
    beta_port = float((weights * beta_vec).sum())
    norm = float((beta_vec**2).sum())
    if norm <= 1e-8:
        return weights
    return weights - beta_vec * (beta_port / norm)


def run_backtest(
    preds: pd.DataFrame,
    ff: pd.DataFrame,
    factors: list[str],
    cfg_port: dict[str, Any],
    cfg_costs: dict[str, Any],
    bt_cfg: BacktestConfig,
    log_policy_path: Path | None = None,
) -> pd.DataFrame:
    idx = preds.index
    prev_w = pd.Series(0.0, index=factors)
    rows: list[dict[str, Any]] = []
    market_factor = "Mkt_RF" if "Mkt_RF" in factors else None
    try:
        benchmark_series = load_benchmark_series(bt_cfg.benchmark, ff, idx)
        if not benchmark_series.index.equals(pd.Index(idx)):
            raise ValueError("misaligned benchmark index")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise BacktestError("Benchmark data missing date alignment") from exc

    for t in idx:
        factor_hist = ff.loc[:t, factors].dropna()
        hist_slice = factor_hist if len(factor_hist) > 0 else None
        risk_cov = (
            risk_model(factor_hist, halflife=bt_cfg.risk_halflife)
            if len(factor_hist) > 1
            else None
        )

        mu = preds.loc[t, factors]
        w, _turnover, score = build_weights(
            mu,
            hist_slice,
            cfg_port,
            prev_w=prev_w,
            risk_cov=risk_cov,
        )
        w_ridge_series = pd.Series(w).reindex(factors).fillna(0.0)
        try:
            hybrid = blend_with_llm(
                date=t,
                w_ridge=w_ridge_series,
                beta_neutral=bt_cfg.beta_neutral,
                risk_cov=risk_cov,
                market_factor=market_factor,
            )
            w_series = hybrid.w_final.reindex(factors).fillna(0.0)
            if log_policy_path is not None:
                append_policy_log(hybrid, log_policy_path)
        except Exception:
            w_series = w_ridge_series

        if not is_rebalance_date(t, bt_cfg.rebalance_freq):
            adj_w = prev_w.copy()
            delta = adj_w - prev_w
            traded = False
        else:
            adj_w, delta, traded = apply_trade_threshold(
                w_series, prev_w, bt_cfg.trade_eps
            )
            adj_w = enforce_leverage_constraints(
                adj_w, bt_cfg.borrow_cap, bt_cfg.floor_cash
            )
            delta = adj_w - prev_w
            if bt_cfg.beta_neutral and market_factor:
                adj_w = _project_beta_neutral(adj_w, risk_cov, market_factor)
                adj_w = enforce_leverage_constraints(
                    adj_w, bt_cfg.borrow_cap, bt_cfg.floor_cash
                )
                delta = adj_w - prev_w

        r = ff.loc[t, factors]
        rf_raw = float(ff.loc[t, bt_cfg.rf_col]) if bt_cfg.rf_col in ff.columns else 0.0
        rf_monthly = rf_raw / 1200.0
        port_gross = rf_monthly + float((adj_w * r).sum())
        slippage = compute_slippage(cfg_costs, delta, hist_slice)
        try:
            turnover = float(np.abs(delta).sum())
            if turnover < -1e-9:
                raise ValueError("negative turnover")
        except Exception as exc:  # pragma: no cover - defensive guard
            raise BacktestError("Negative turnover detected") from exc
        transaction_cost = slippage + turnover * bt_cfg.cost_bps * 1e-4
        pnl_gross = port_gross
        pnl_net = pnl_gross - transaction_cost
        net_excess = pnl_net - rf_monthly
        benchmark_ret = (
            float(benchmark_series.loc[t]) if t in benchmark_series.index else 0.0
        )

        rows.append(
            {
                "date": t,
                "rf": rf_monthly,
                "gross_ret": pnl_gross,
                "slippage": slippage,
                "transaction_cost": transaction_cost,
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
                "net_ret": pnl_net,
                "net_excess": net_excess,
                "benchmark_ret": benchmark_ret,
                "turnover": turnover,
                "trade_mask": bool(traded and (delta.abs() > 0).any()),
                **{f"ret_{f}": float(r[f]) for f in factors},
                **{f"w_{f}": float(adj_w.get(f, 0.0)) for f in factors},
                **{f"s_{f}": float(score.get(f, 0.0)) for f in factors},
            }
        )

        prev_w = adj_w

    return pd.DataFrame(rows).set_index("date")


def normalize_costs(cfg_costs: dict[str, Any]) -> dict[str, Any]:
    costs = dict(cfg_costs)
    if "fixed_bps" not in costs and costs.get("bps_per_turnover") is not None:
        costs["fixed_bps"] = costs.get("bps_per_turnover", 0.0)
    model_value = costs.get("slippage_model") or costs.get("model")
    if isinstance(model_value, SlippageModel):
        model = model_value
    else:
        model_str = str(model_value or SlippageModel.FIXED.value)
        if model_str.lower().startswith("slippagemodel."):
            model_str = model_str.split(".", 1)[1]
        model = SlippageModel(model_str.lower())
    costs["model"] = model.value
    if model == SlippageModel.FIXED and costs.get("fixed_bps") is None:
        costs["fixed_bps"] = 0.0
    if model == SlippageModel.VOL and costs.get("vol_k") is None:
        raise ValueError("vol_k must be provided when slippage model is 'vol'.")
    if model == SlippageModel.SPREAD and costs.get("spread_bps") is None:
        raise ValueError("spread_bps must be provided when slippage model is 'spread'.")
    return costs


def compute_ic(preds: pd.DataFrame, ff: pd.DataFrame, factor: str) -> float:
    realized = ff[factor].shift(-1).reindex(preds.index)
    aligned = preds[factor]
    mask = aligned.notna() & realized.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(aligned[mask].corr(realized[mask], method="pearson"))


def factor_ablation(
    output_dir: Path,
    base_preds: pd.DataFrame,
    ff: pd.DataFrame,
    factors: list[str],
    cfg_port: dict[str, Any],
    cfg_costs: dict[str, Any],
    bt_cfg: BacktestConfig,
    base_sharpe: float,
    base_ic: dict[str, float],
) -> None:
    summaries: list[dict[str, float]] = []

    for i, factor in enumerate(factors):
        perm_preds = base_preds.copy()
        rng = np.random.default_rng(DEFAULT_SEED + i)
        perm_preds[factor] = rng.permutation(perm_preds[factor].to_numpy())

        perm_bt = run_backtest(
            perm_preds,
            ff,
            factors,
            cfg_port,
            cfg_costs,
            bt_cfg,
            # For ablation, keep regimes fixed or permute? Keeping fixed/None.
        )
        perm_sharpe = sharpe_excess(perm_bt["net_excess"])
        perm_ic = compute_ic(perm_preds, ff, factor)
        summaries.append(
            {
                "factor": factor,
                "delta_sharpe": perm_sharpe - base_sharpe,
                "delta_ic": perm_ic - base_ic.get(factor, np.nan),
            }
        )

    output_path = output_dir / "ablation_summary.csv"
    pd.DataFrame(summaries).to_csv(output_path, index=False)
    print(f"Saved ablation summary -> {output_path}")


def save_turnover_rolling(output_dir: Path, bt: pd.DataFrame) -> None:
    rolling = bt["turnover"].rolling(window=12, min_periods=1).mean()
    output_path = output_dir / "turnover_rolling.parquet"
    rolling.to_frame(name="turnover_rolling").to_parquet(output_path)
    print(f"Saved turnover rolling metrics -> {output_path}")


def perform_hyperparam_sweep(
    config: ConfigModel,
    output_dir: Path,
    config_path: Path,
    ff: pd.DataFrame,
    factors: list[str],
    cfg_port: dict[str, Any],
    cfg_costs: dict[str, Any],
    bt_cfg: BacktestConfig,
) -> None:
    alpha_grid = config.model.ridge_alpha_grid or [0.5, 1.0, 3.0, 5.0]
    temp_grid = config.portfolio.temperature_grid or [0.3, 0.5, 1.0]

    results: list[dict[str, float]] = []

    for alpha in alpha_grid:
        preds, _ = train_models(
            ridge_alpha=alpha,
            capture_coeffs=False,
            save_outputs=False,
            config_path=config_path,
        )
        preds = preds.reindex(ff.index)
        preds = preds[[*factors, bt_cfg.rf_col]].dropna()

        for temp in temp_grid:
            port_cfg = dict(cfg_port)
            port_cfg["temperature"] = temp

            bt = run_backtest(
                preds,
                ff,
                factors,
                port_cfg,
                cfg_costs,
                bt_cfg,
            )
            sharpe = sharpe_excess(bt["net_excess"])
            ann_return, ann_sigma = annualize_monthly(bt["net_ret"])
            mdd = max_drawdown(bt["net_ret"])

            results.append(
                {
                    "ridge_alpha": alpha,
                    "temperature": temp,
                    "sharpe": sharpe,
                    "ann_ret": ann_return,
                    "ann_vol": ann_sigma,
                    "max_dd": float(mdd),
                }
            )

    if not results:
        return

    summary = pd.DataFrame(results)
    cleaned = summary["sharpe"].replace([np.inf, -np.inf], np.nan)
    if cleaned.notna().any():
        best_idx = cleaned.idxmax()
    else:
        best_idx = summary.index[0]
    summary["is_best"] = False
    summary.loc[best_idx, "is_best"] = True
    output = output_dir / "sweep_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Saved hyperparameter sweep summary -> {output}")


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config_path = Path(config_path)
    config = load_config(config_path)
    processed_dir = Path(config.data.out_processed)
    raw_dir = Path(config.data.out_raw)

    set_global_seed(DEFAULT_SEED)

    preds, coeff_frames = train_models(
        capture_coeffs=True, save_outputs=True, config_path=Path(config_path)
    )
    plot_ridge_coeffs(config, coeff_frames)

    ff = pd.read_parquet(raw_dir / "ff" / "ff_monthly.parquet")
    ff.index = pd.to_datetime(ff.index)

    preds = preds.reindex(ff.index).dropna()
    factors = [
        col
        for col in ["Mkt_RF", "SMB", "HML", "UMD"]
        if col in preds.columns and col in ff.columns
    ]
    if not factors:
        raise ValueError("No overlapping factor columns found in preds vs ff.")

    cfg_port = config.portfolio.model_dump(by_alias=True)
    cfg_costs = normalize_costs(config.costs.model_dump(by_alias=True))
    bt_cfg = BacktestConfig.from_model(config)

    base_bt = run_backtest(
        preds,
        ff,
        factors,
        cfg_port,
        cfg_costs,
        bt_cfg,
        log_policy_path=processed_dir / "llm_policy_log.csv",
    )
    base_path = processed_dir / "backtest.parquet"
    base_bt.to_parquet(base_path)
    print(f"Saved backtest results -> {base_path}")

    save_turnover_rolling(processed_dir, base_bt)
    generate_performance_artifacts(base_bt, ff, processed_dir, config)

    # --- Tradeable ETF Backtest ---
    # Apply the same signals (preds) to tradeable ETF returns
    etf_panel_path = processed_dir / "panel_etfs.parquet"
    if etf_panel_path.exists():
        etf_panel = pd.read_parquet(etf_panel_path)
        # Mapping: ETF Name -> Factor Name
        # Based on macrotones/features/dataset_etfs.py
        mapping = {
            "Value": "HML",
            "Size": "SMB",
            "Momentum": "UMD",
            "Market": "Mkt_RF"
        }
        # Select and rename available columns
        available_etfs = [c for c in mapping if c in etf_panel.columns]
        if available_etfs:
            etf_returns = etf_panel[available_etfs].rename(columns=mapping)
            # Ensure RF exists (use FF RF if available, else 0)
            if "RF" not in etf_returns.columns:
                if "RF" in ff.columns:
                    etf_returns["RF"] = ff["RF"].reindex(etf_returns.index).fillna(0.0)
                else:
                    etf_returns["RF"] = 0.0
            
            # Align preds to etf_returns
            etf_preds = preds.reindex(etf_returns.index).dropna()
            
            # Use only factors present in both
            etf_factors = [f for f in factors if f in etf_returns.columns]
            
            if etf_factors:
                etf_bt = run_backtest(
                    etf_preds,
                    etf_returns,
                    etf_factors,
                    cfg_port,
                    cfg_costs,
                    bt_cfg,
                    log_policy_path=None
                )
                etf_out_path = processed_dir / "backtest_etfs.parquet"
                etf_bt.to_parquet(etf_out_path)
                print(f"Saved ETF backtest results -> {etf_out_path}")
            else:
                print("Warning: No matching factors for ETF backtest.")
    else:
        print(
            f"Warning: ETF panel not found at {etf_panel_path}. Skipping ETF backtest."
        )

    base_sharpe = sharpe_excess(base_bt["net_excess"])
    base_ic = {factor: compute_ic(preds, ff, factor) for factor in factors}
    factor_ablation(
        processed_dir,
        preds,
        ff,
        factors,
        cfg_port,
        cfg_costs,
        bt_cfg,
        base_sharpe,
        base_ic,
    )

    perform_hyperparam_sweep(
        config,
        processed_dir,
        Path(config_path),
        ff,
        factors,
        cfg_port,
        cfg_costs,
        bt_cfg,
    )

    ann_return, ann_sigma = annualize_monthly(base_bt["net_ret"])
    sharpe = base_sharpe
    mdd = max_drawdown(base_bt["net_ret"])

    print(f"  AnnRet: {ann_return:0.4f}")
    print(f"  AnnVol: {ann_sigma:0.4f}")
    print(f"  Sharpe_excess: {sharpe:0.4f}")
    print(f"  MaxDD: {mdd:0.4f}")


if __name__ == "__main__":
    main()
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
