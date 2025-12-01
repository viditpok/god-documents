"""Entry point for running the stat-arb project pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path

from src.config import ProjectConfig
from src.pipeline import StatArbPipeline

# Suppress expected warnings from pandas/numpy operations on NaN values
# These warnings occur when computing statistics on data with missing values, which is expected
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")


def main() -> None:
    config = ProjectConfig(
        price_path=Path("data/coin_all_prices_full.csv"),
        universe_path=Path("data/coin_universe_150K_40.csv"),
    )
    pipeline = StatArbPipeline(config)
    result = pipeline.run()
    pipeline.save_outputs(result)
    print("Sharpe ratio:", result.sharpe)
    print("Max drawdown:", result.max_drawdown)
    print("Outputs saved under:", config.output_dir)


if __name__ == "__main__":
    main()

