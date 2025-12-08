from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from macrotones.backtest.metrics import (
    annualize_monthly,
    max_drawdown,
    sharpe_excess,
)
from macrotones.backtest.utils import (
    apply_trade_threshold,
    compute_slippage,
    enforce_leverage_constraints,
    is_rebalance_date,
)
from macrotones.config.schema import SlippageModel
from macrotones.portfolio.allocators import build_weights
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed


def parse_grid(raw: str) -> list[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-bps",
        type=str,
        default="0,5,10,25",
        help="Comma-separated list of fixed slippage bps to evaluate.",
    )
    parser.add_argument(
        "--costs",
        type=str,
        default=None,
        help="Alias for --fixed-bps (comma-separated list of fixed slippage bps).",
    )
    parser.add_argument(
        "--vol-k",
        type=str,
        default="0.5,1.0",
        help="Comma-separated list of k multipliers for volatility slippage.",
    )
    parser.add_argument(
        "--spread-bps",
        type=str,
        default="5,10",
        help="Comma-separated list of spread proxy bps to evaluate.",
    )
    args = parser.parse_args()

    set_global_seed(DEFAULT_SEED)

    cfg: dict[str, Any] = yaml.safe_load(open("config/project.yaml"))
    pro = Path(cfg["data"]["out_processed"])
    raw = Path(cfg["data"]["out_raw"])

    preds = pd.read_parquet(pro / "preds.parquet")
    ff = pd.read_parquet(raw / "ff" / "ff_monthly.parquet")

    preds.index = pd.to_datetime(preds.index)
    ff.index = pd.to_datetime(ff.index)

    factors = [
        c
        for c in ["Mkt_RF", "SMB", "HML", "UMD"]
        if c in preds.columns and c in ff.columns
    ]
    if not factors:
        print("No overlapping factor columns found.")
        return

    fixed_source = args.costs if args.costs else args.fixed_bps
    fixed_grid = parse_grid(fixed_source)
    vol_grid = parse_grid(args.vol_k)
    spread_grid = parse_grid(args.spread_bps)

    cfg_port = cfg.get("portfolio", {}).copy()
    cfg_costs_base = dict(cfg.get("costs", {}))
    if "fixed_bps" not in cfg_costs_base and "bps_per_turnover" in cfg_costs_base:
        cfg_costs_base["fixed_bps"] = cfg_costs_base.get("bps_per_turnover", 0.0)
    cfg_costs_base.setdefault(
        "model",
        cfg_costs_base.get("slippage_model", SlippageModel.FIXED.value),
    )
    rebalance_freq = cfg.get("project", {}).get("rebalance", "M")
    trade_eps = float(cfg_port.get("trade_threshold", 0.0))
    borrow_cap = cfg_port.get("borrow_cap")
    floor_cash = cfg_port.get("floor_cash")

    scenarios: list[tuple[SlippageModel, float]] = []
    scenarios.extend((SlippageModel.FIXED, val) for val in fixed_grid)
    scenarios.extend((SlippageModel.VOL, val) for val in vol_grid)
    scenarios.extend((SlippageModel.SPREAD, val) for val in spread_grid)
    if not scenarios:
        default_param = float(cfg_costs_base.get("fixed_bps", 0.0))
        scenarios.append((SlippageModel.FIXED, default_param))

    results = []

    for model, param in scenarios:
        costs_cfg = cfg_costs_base.copy()
        costs_cfg["model"] = model.value
        if model == SlippageModel.FIXED:
            costs_cfg["fixed_bps"] = param
        elif model == SlippageModel.VOL:
            costs_cfg["vol_k"] = param
        elif model == SlippageModel.SPREAD:
            costs_cfg["spread_bps"] = param

        net_returns = []
        prev_w = pd.Series(0.0, index=factors)

        for t in preds.index:
            hist_slice = ff.loc[:t].dropna()
            if len(hist_slice) > 0:
                hist_slice = hist_slice[factors]
            else:
                hist_slice = None

            mu = preds.loc[t, factors]
            r = ff.loc[t, factors]
            rf_raw = float(ff.loc[t, "RF"]) if "RF" in ff.columns else 0.0
            rf_monthly = rf_raw / 1200.0

            w, _turnover, _score = build_weights(
                mu, hist_slice, cfg_port, prev_w=prev_w
            )
            w_series = pd.Series(w).reindex(factors).fillna(0.0)

            if not is_rebalance_date(t, rebalance_freq):
                adj_w = prev_w.copy()
                delta = adj_w - prev_w
            else:
                adj_w, delta, _ = apply_trade_threshold(w_series, prev_w, trade_eps)
                adj_w = enforce_leverage_constraints(adj_w, borrow_cap, floor_cash)
                delta = adj_w - prev_w

            port_gross = rf_monthly + float((adj_w * r).sum())
            slip = compute_slippage(costs_cfg, delta, hist_slice)
            net_returns.append(port_gross - slip)
            prev_w = adj_w

        port_series = pd.Series(net_returns, index=preds.index)
        if "RF" in ff.columns:
            rf_series = ff["RF"].reindex(preds.index).fillna(0.0) / 1200.0
        else:
            rf_series = pd.Series(0.0, index=preds.index)

        ann_return, ann_volatility = annualize_monthly(port_series)
        shr = sharpe_excess(port_series - rf_series)
        dd = max_drawdown(port_series)

        results.append(
            {
                "model": model.value,
                "parameter": param,
                "ann_ret": ann_return,
                "ann_vol": ann_volatility,
                "sharpe_excess": shr,
                "max_dd": dd,
            }
        )

        print(
            f"Model {model.value:7s} param {param:6.2f}: "
            f"Sharpe {shr:.3f}, Return {ann_return:.3f}, Vol {ann_volatility:.3f}"
        )

    # Save results
    df_results = pd.DataFrame(results)
    out_path = pro / "costs_sweep.csv"
    df_results.to_csv(out_path, index=False)

    print(f"Saved cost sweep results to: {out_path}")


if __name__ == "__main__":
    main()
