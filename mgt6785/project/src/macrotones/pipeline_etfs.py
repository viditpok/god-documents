from pathlib import Path

import pandas as pd

from macrotones.backtest.config import BacktestConfig
from macrotones.backtest.engine import normalize_costs, run_backtest
from macrotones.backtest.metrics import annualize_monthly, max_drawdown, sharpe_excess
from macrotones.config.schema import load_config
from macrotones.data import fetch_etfs
from macrotones.features import dataset_etfs
from macrotones.models.train import train_models

# Configuration
CONFIG_PATH = Path("config/project.yaml")
ETF_AGNOSTIC_TARGETS = ["Value", "Size", "Momentum", "Quality", "LowVol", "Growth"]


def main():
    print("\n🚀 Starting ETF Pipeline...\n")

    # 1. Fetch Data
    print("--- 1. Fetching ETF Data ---")
    fetch_etfs.main()

    # 2. Build Dataset
    print("\n--- 2. Building ETF Panel ---")
    dataset_etfs.main()

    # 3. Train Models
    print("\n--- 3. Training Models (Ridge/XGBoost) ---")
    # This saves to data/processed/preds_etfs.parquet due to our earlier refactor
    preds, _ = train_models(
        panel_path="panel_etfs.parquet", targets=ETF_AGNOSTIC_TARGETS, save_outputs=True
    )

    # 4. Backtest
    print("\n--- 4. Running Backtest ---")
    config = load_config(CONFIG_PATH)
    processed_dir = Path(config.data.out_processed)

    # Load ETF Returns (acting as both "Features" and "Tradable Assets")
    # We use the raw ETF returns for PnL calculation
    etfs = pd.read_parquet(Path(config.data.out_raw) / "etfs/etfs_monthly.parquet")

    # Needs to align timestamps
    preds = preds.reindex(etfs.index).dropna()
    common_idx = preds.index.intersection(etfs.index)
    preds = preds.loc[common_idx]
    etfs = etfs.loc[common_idx]

    # Setup Backtest Config
    cfg_port = config.portfolio.model_dump(by_alias=True)
    cfg_costs = normalize_costs(config.costs.model_dump(by_alias=True))
    bt_cfg = BacktestConfig.from_model(config)

    # Run
    # Note: 'etfs' dataframe has the returns for the factors we are trading.
    bt_results = run_backtest(
        preds, etfs, ETF_AGNOSTIC_TARGETS, cfg_port, cfg_costs, bt_cfg
    )

    # 5. Save Results
    out_path = processed_dir / "backtest_etfs.parquet"
    bt_results.to_parquet(out_path)
    print(f"✅ Saved ETF Backtest Results -> {out_path}")

    # 6. Print Metrics
    ann_ret, ann_vol = annualize_monthly(bt_results["net_ret"])
    sharpe = sharpe_excess(bt_results["net_excess"])
    mdd = max_drawdown(bt_results["net_ret"])

    print("\n📊 ETF Strategy Performance:")
    print(f"  Ann Return: {ann_ret:.2%}")
    print(f"  Ann Vol:    {ann_vol:.2%}")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  Max DD:     {mdd:.2%}")


if __name__ == "__main__":
    main()
