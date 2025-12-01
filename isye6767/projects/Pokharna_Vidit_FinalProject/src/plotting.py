"""Plotting utilities for deliverables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotBuilder:
    """Encapsulates matplotlib plotting routines."""

    def __init__(self) -> None:
        sns.set_style("whitegrid")

    @staticmethod
    def plot_cumulative_returns(
        factor_returns: pd.DataFrame,
        price_returns: pd.DataFrame,
        output_path: Path,
    ) -> None:
        # Combine all returns into a single dataframe
        all_returns = pd.DataFrame(index=factor_returns.index)
        for column in factor_returns.columns:
            all_returns[column] = factor_returns[column]
        for column in price_returns.columns:
            all_returns[column] = price_returns[column]
        
        # Compute cumulative returns
        compounded = (1 + all_returns.fillna(0)).cumprod() - 1

        # Define expected columns with very distinct colors and line styles
        expected_columns = ["factor_1", "factor_2", "BTC", "ETH"]
        # Use very distinct colors: Blue, Orange, Green, Red with different line styles
        colors = ["#0066CC", "#FF6600", "#00AA00", "#CC0000"]  # Bright Blue, Bright Orange, Bright Green, Bright Red
        linestyles = ["-", "-", "-", "-"]  # All solid lines
        linewidths = [2.0, 2.0, 2.0, 2.0]  # All same width for visibility
        
        plt.figure(figsize=(14, 7))
        
        # Plot each expected column explicitly with distinct colors
        plotted_count = 0
        for idx, col in enumerate(expected_columns):
            if col in compounded.columns:
                # Check if there's any non-zero data
                if not compounded[col].isna().all():
                    valid_data = compounded[col].dropna()
                    plt.plot(
                        compounded.index, 
                        compounded[col], 
                        label=col, 
                        linewidth=linewidths[idx],
                        color=colors[idx],
                        linestyle=linestyles[idx],
                        alpha=0.9
                    )
                    plotted_count += 1
                    # Debug: print range of values
                    if len(valid_data) > 0:
                        print(f"    {col}: min={valid_data.min():.6f}, max={valid_data.max():.6f}, mean={valid_data.mean():.6f}, color={colors[idx]}")
                else:
                    print(f"Warning: {col} has no valid data to plot")
            else:
                print(f"Warning: {col} not found in compounded returns")
        
        # Also plot any other columns that might exist
        for col in compounded.columns:
            if col not in expected_columns:
                plt.plot(compounded.index, compounded[col], label=col, linewidth=1.5, linestyle="--", alpha=0.7)
        
        plt.title("Cumulative Returns: Eigen-portfolios vs BTC/ETH", fontsize=14, fontweight='bold')
        plt.ylabel("Return", fontsize=12)
        plt.xlabel("Time", fontsize=12)
        # Ensure legend shows all items with distinct colors
        legend = plt.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, ncol=2)
        # Make sure legend colors match plot colors
        for idx, (text, line) in enumerate(zip(legend.get_texts(), legend.get_lines())):
            if idx < len(colors):
                line.set_color(colors[idx])
                line.set_linewidth(2.0)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plotted {plotted_count} out of {len(expected_columns)} expected series")

    @staticmethod
    def plot_sorted_weights(weights: pd.Series, timestamp: pd.Timestamp, output_path: Path) -> None:
        sorted_weights = weights.sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sorted_weights.plot(kind="bar")
        plt.title(f"Eigen-portfolio Weights at {timestamp.isoformat()}")
        plt.ylabel("Weight")
        plt.xlabel("Token")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_s_score(series: pd.Series, token: str, output_path: Path) -> None:
        plt.figure(figsize=(12, 4))
        plt.plot(series.index, series.values)
        plt.title(f"s-score evolution for {token}")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("s-score")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_strategy_performance(
        equity_curve: pd.Series, initial_capital: float, output_path: Path
    ) -> None:
        cumulative_return = (equity_curve - initial_capital) / initial_capital
        plt.figure(figsize=(12, 4))
        plt.plot(equity_curve.index, cumulative_return)
        plt.title("Strategy Cumulative Return")
        plt.ylabel("Return")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_return_histogram(returns: pd.Series, output_path: Path) -> None:
        plt.figure(figsize=(8, 4))
        sns.histplot(returns.dropna(), bins=40, kde=True)
        plt.title("Histogram of Hourly Strategy Returns")
        plt.xlabel("Return")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
