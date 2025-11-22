import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_time_grid(path: Path) -> pd.Series:
    grid = pd.read_csv(path)
    return grid["time"]


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_paths(
    time_grid: pd.Series,
    matrix: pd.DataFrame,
    sample: int,
    title: str,
    ylabel: str,
    output: Path,
):
    plt.figure(figsize=(11, 6))
    subset = matrix.head(sample)
    x = time_grid.to_numpy()
    for _, row in subset.iterrows():
        values = row.iloc[1:].to_numpy(dtype=float)
        plt.plot(x, values, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_histogram(values, title: str, xlabel: str, output: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Task 1 CSV outputs.")
    parser.add_argument("--outputs", default="outputs", help="Directory containing Task 1 CSV files.")
    parser.add_argument(
        "--plots", default="plots", help="Directory where PNG files will be written."
    )
    parser.add_argument(
        "--paths", type=int, default=100, help="Number of simulated paths to plot."
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    time_grid = load_time_grid(outputs_dir / "time_grid.csv")
    stock_df = load_matrix(outputs_dir / "stock_paths.csv")
    option_df = load_matrix(outputs_dir / "option_prices.csv")
    hedging_df = load_matrix(outputs_dir / "hedging_errors.csv")

    plot_paths(
        time_grid,
        stock_df,
        args.paths,
        "Simulated Stock Paths",
        "Stock Price",
        plots_dir / "stock_paths.png",
    )
    plot_paths(
        time_grid,
        option_df,
        args.paths,
        "Simulated Option Prices",
        "Option Price",
        plots_dir / "option_prices.png",
    )

    final_errors = hedging_df.iloc[:, -1].to_numpy(dtype=float)
    plot_histogram(
        final_errors,
        "Final Hedging Error Distribution",
        "Hedging Error",
        plots_dir / "hedging_error.png",
    )
    print(f"Task 1 plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
