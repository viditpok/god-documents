"""Main orchestration pipeline for the stat-arb project."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_loader import MarketDataLoader
from .metrics import max_drawdown, sharpe_ratio
from .ou_model import OUParams, OUProcessEstimator
from .pca_model import PCAFactorModel, PCAResult
from .plotting import PlotBuilder
from .portfolio import PortfolioSimulator
from .signals import SignalGenerator


@dataclass
class PipelineResult:
    """Aggregate of all outputs."""

    eigenvectors: Dict[int, pd.DataFrame]
    factor_returns: pd.DataFrame
    s_scores: pd.DataFrame
    trading_signals: pd.DataFrame
    portfolio_history: pd.DataFrame
    sharpe: float
    max_drawdown: float


class StatArbPipeline:
    """Runs computations for all project tasks."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.data_loader = MarketDataLoader(config)
        self.pca_model = PCAFactorModel(n_factors=2)
        self.ou_estimator = OUProcessEstimator()
        self.signal_generator = SignalGenerator(config.thresholds)
        self.portfolio = PortfolioSimulator(config.initial_capital)
        self.plot_builder = PlotBuilder()

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        self.config.ensure_directories()

        timeline = list(
            self.data_loader.iter_time_range(self.config.testing_start, self.config.testing_end)
        )
        all_tokens = list(self.data_loader.price_df.columns)
        eigenvectors: Dict[int, List[pd.Series]] = {1: [], 2: []}
        eigen_index: List[pd.Timestamp] = []
        factor_records: List[Dict[str, float]] = []
        s_score_rows: List[pd.Series] = []
        signal_records: List[pd.Series] = []
        token_universe: Set[str] = set()

        portfolio_states: List[Dict[str, float]] = []
        last_equity: Optional[float] = None

        for timestamp in timeline:
            window_start = timestamp - timedelta(hours=self.config.window_hours)
            window_end = timestamp - timedelta(hours=1)
            if window_start < self.data_loader.price_df.index[0]:
                continue

            tokens = self.data_loader.get_universe_tokens(timestamp)
            if self.config.required_tokens:
                tokens = sorted(set(tokens).union(self.config.required_tokens))
            if not tokens:
                continue
            token_universe.update(tokens)
            window_returns = self.data_loader.get_return_window(window_start, window_end, tokens)
            if window_returns.empty or len(window_returns) < self.config.window_hours:
                continue

            coverage_counts = window_returns.notna().sum()
            sufficient = coverage_counts[coverage_counts >= self.config.coverage_ratio * len(window_returns)]
            window_returns = window_returns.loc[:, sufficient.index]
            window_returns = window_returns.dropna(axis=0, how="any")

            if window_returns.shape[1] < max(self.config.min_tokens, 2):
                continue

            pca_result = self.pca_model.compute(window_returns)
            if pca_result is None:
                continue

            eigen_index.append(timestamp)
            weight_row_1 = pd.Series(0.0, index=all_tokens)
            weight_row_2 = pd.Series(0.0, index=all_tokens)
            weight_row_1.update(pca_result.weight_vectors.get(1, pd.Series(dtype=float)))
            weight_row_2.update(pca_result.weight_vectors.get(2, pd.Series(dtype=float)))
            eigenvectors[1].append(weight_row_1)
            eigenvectors[2].append(weight_row_2)

            # Factor returns for plotting - compute using eigenportfolio weights and actual returns at time t
            # F_{j, t} = sum(Q^{(j)} * R_t) where R_t are the actual returns at time t
            # Note: returns_df[t] represents return from t-1 to t
            returns_at_t = self.data_loader.get_return_window(timestamp, timestamp, tokens)
            factor_1_return = 0.0
            factor_2_return = 0.0
            if not returns_at_t.empty and timestamp in returns_at_t.index:
                for token in pca_result.tokens:
                    if token in returns_at_t.columns and token in weight_row_1.index:
                        ret = returns_at_t.loc[timestamp, token]
                        if not pd.isna(ret):
                            weight_1 = weight_row_1[token]
                            weight_2 = weight_row_2[token]
                            if not pd.isna(weight_1) and weight_1 != 0:
                                factor_1_return += weight_1 * ret
                            if not pd.isna(weight_2) and weight_2 != 0:
                                factor_2_return += weight_2 * ret
            
            factor_records.append(
                {
                    "timestamp": timestamp,
                    "factor_1": factor_1_return,
                    "factor_2": factor_2_return,
                }
            )

            # Regression residuals and OU estimates
            params_by_token: Dict[str, OUParams] = {}
            factor_window = pca_result.factor_window

            for token in pca_result.tokens:
                y = pca_result.window_returns[token].values
                design = np.column_stack(
                    [
                        np.ones_like(y),
                        factor_window["F1"].values,
                        factor_window["F2"].values,
                    ]
                )
                coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                residuals = y - design @ coeffs
                params = self.ou_estimator.estimate(residuals)
                if params:
                    params_by_token[token] = params

            if not params_by_token:
                continue

            avg_a = float(np.mean([p.a for p in params_by_token.values()]))
            s_row = pd.Series(index=all_tokens, dtype=float)
            for token, params in params_by_token.items():
                s = self.ou_estimator.compute_s_score(params, avg_a)
                params.s_score = s
                s_row[token] = s
            s_score_rows.append(s_row)

            # Trading signal evaluation
            signal_row = pd.Series(index=all_tokens, dtype=object)
            prices = self.data_loader.get_prices(timestamp, tokens)
            for token in tokens:
                position = self.portfolio.positions.get(token, 0)
                s_score = s_row.get(token)
                decision = self.signal_generator.evaluate(s_score, position)
                price = prices.get(token)
                if decision.action in {"BUY_OPEN", "SELL_OPEN", "CLOSE_LONG", "CLOSE_SHORT"}:
                    if price is not None and not pd.isna(price):
                        self.portfolio.apply_trade(token, decision.action, float(price))
                signal_row[token] = decision.action
            signal_records.append(signal_row)

            # Portfolio valuation
            price_slice = self.data_loader.price_df.loc[timestamp]
            holdings_value = self.portfolio.holdings_value(price_slice)
            equity = self.portfolio.cash + holdings_value
            hourly_return = 0.0
            if last_equity is not None and last_equity != 0:
                hourly_return = (equity - last_equity) / last_equity
            last_equity = equity
            portfolio_states.append(
                {
                    "timestamp": timestamp,
                    "cash": self.portfolio.cash,
                    "holdings_value": holdings_value,
                    "equity": equity,
                    "hourly_return": hourly_return,
                }
            )

        eigenvector_frames = {
            idx: pd.DataFrame(eigenvectors[idx], index=eigen_index) for idx in eigenvectors
        }
        factor_df = pd.DataFrame(factor_records).set_index("timestamp")
        s_scores_df = pd.DataFrame(s_score_rows, index=eigen_index)
        signals_df = pd.DataFrame(signal_records, index=eigen_index)
        signals_df = signals_df.loc[:, sorted(token_universe)]
        portfolio_history = pd.DataFrame(portfolio_states).set_index("timestamp")

        sharpe = sharpe_ratio(portfolio_history["hourly_return"])
        mdd = max_drawdown(portfolio_history["equity"])

        return PipelineResult(
            eigenvectors=eigenvector_frames,
            factor_returns=factor_df,
            s_scores=s_scores_df,
            trading_signals=signals_df,
            portfolio_history=portfolio_history,
            sharpe=sharpe,
            max_drawdown=mdd,
        )

    # Output helpers -----------------------------------------------------

    def save_outputs(self, result: PipelineResult) -> None:
        """Persist CSVs and figures."""
        # Task 1 CSVs
        result.eigenvectors[1].to_csv(self.config.csv_dir / "task1a_1.csv", float_format="%.8f")
        result.eigenvectors[2].to_csv(self.config.csv_dir / "task1a_2.csv", float_format="%.8f")

        # Task 4 trading signals CSV
        result.trading_signals.to_csv(self.config.csv_dir / "trading_signal.csv")

        # Figures
        # Get BTC and ETH returns, ensuring they align with factor returns index
        btc_eth_returns = self.data_loader.returns_df[["BTC", "ETH"]].reindex(
            result.factor_returns.index
        )
        
        # Debug: Check data availability
        factor_cols = ["factor_1", "factor_2"]
        print(f"  Factor returns shape: {result.factor_returns[factor_cols].shape}")
        print(f"  BTC/ETH returns shape: {btc_eth_returns.shape}")
        print(f"  Factor returns non-null counts: {result.factor_returns[factor_cols].notna().sum().to_dict()}")
        print(f"  BTC/ETH returns non-null counts: {btc_eth_returns.notna().sum().to_dict()}")
        
        try:
            self.plot_builder.plot_cumulative_returns(
                result.factor_returns[factor_cols],
                btc_eth_returns,
                self.config.figures_dir / "task1b_cumulative_returns.png",
            )
            print(f"✓ Generated: task1b_cumulative_returns.png")
        except Exception as e:
            print(f"✗ Error generating task1b plot: {e}")
            import traceback
            traceback.print_exc()

        # Task 2: Plot eigenportfolio 1 at first timestamp, eigenportfolio 2 at second timestamp
        ts1 = pd.Timestamp("2021-09-26T12:00:00Z")
        ts2 = pd.Timestamp("2022-04-15T20:00:00Z")
        
        # Plot eigenportfolio 1 at timestamp 1
        if ts1 in result.eigenvectors[1].index:
            try:
                self.plot_builder.plot_sorted_weights(
                    result.eigenvectors[1].loc[ts1].dropna(),
                    ts1,
                    self.config.figures_dir / f"task2_weights_1_{ts1.strftime('%Y%m%d%H%M')}.png",
                )
                print(f"✓ Generated: task2_weights_1_{ts1.strftime('%Y%m%d%H%M')}.png")
            except Exception as e:
                print(f"✗ Error generating task2 plot for eigenportfolio 1 at {ts1}: {e}")
        
        # Plot eigenportfolio 2 at timestamp 2
        if ts2 in result.eigenvectors[2].index:
            try:
                self.plot_builder.plot_sorted_weights(
                    result.eigenvectors[2].loc[ts2].dropna(),
                    ts2,
                    self.config.figures_dir / f"task2_weights_2_{ts2.strftime('%Y%m%d%H%M')}.png",
                )
                print(f"✓ Generated: task2_weights_2_{ts2.strftime('%Y%m%d%H%M')}.png")
            except Exception as e:
                print(f"✗ Error generating task2 plot for eigenportfolio 2 at {ts2}: {e}")

        s_score_slice = result.s_scores.loc[
            "2021-09-26":"2021-10-25", ["BTC", "ETH"]
        ]
        if "BTC" in s_score_slice.columns:
            try:
                self.plot_builder.plot_s_score(
                    s_score_slice["BTC"].dropna(),
                    "BTC",
                    self.config.figures_dir / "task3_btc_s_score.png",
                )
                print(f"✓ Generated: task3_btc_s_score.png")
            except Exception as e:
                print(f"✗ Error generating task3 BTC plot: {e}")
        if "ETH" in s_score_slice.columns:
            try:
                self.plot_builder.plot_s_score(
                    s_score_slice["ETH"].dropna(),
                    "ETH",
                    self.config.figures_dir / "task3_eth_s_score.png",
                )
                print(f"✓ Generated: task3_eth_s_score.png")
            except Exception as e:
                print(f"✗ Error generating task3 ETH plot: {e}")

        try:
            self.plot_builder.plot_strategy_performance(
                result.portfolio_history["equity"],
                self.config.initial_capital,
                self.config.figures_dir / "cumulative_return.png",
            )
            print(f"✓ Generated: cumulative_return.png")
        except Exception as e:
            print(f"✗ Error generating strategy performance plot: {e}")
        
        try:
            self.plot_builder.plot_return_histogram(
                result.portfolio_history["hourly_return"],
                self.config.figures_dir / "hist_return.png",
            )
            print(f"✓ Generated: hist_return.png")
        except Exception as e:
            print(f"✗ Error generating return histogram: {e}")
