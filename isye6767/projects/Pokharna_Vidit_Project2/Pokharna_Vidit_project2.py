from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Pokharna_Vidit_module1 import (
    DataPreprocessor,
    MarketDataConfig,
    MarketDataManager,
    PriceFeatureEngineer,
    load_tickers,
)
from Pokharna_Vidit_module2 import (
    BacktestEngine,
    ModelPerformance,
    ProbabilitySignalStrategy,
    ReturnModelTrainer,
    TrendFilteredStrategy,
    build_prediction_frame,
    summarize_results,
)
import Pokharna_Vidit_module2 as module2


# pipeline: prepare dataset by loading data, preprocessing, and creating features
def prepare_dataset(
    ticker: str,
    manager: MarketDataManager,
    preprocessor: DataPreprocessor,
    engineer: PriceFeatureEngineer,
) -> pd.DataFrame:
    # data loading
    raw = manager.load(ticker)
    # preprocessing: clean and fill missing values
    cleaned = preprocessor.clean_prices(raw)
    # feature creation: build feature frame with technical indicators
    return engineer.build_feature_frame(cleaned)


# pipeline: evaluate all tickers in universe with training, evaluation, and inference data collection
def evaluate_universe(
    name: str,
    tickers: Iterable[str],
    manager: MarketDataManager,
    preprocessor: DataPreprocessor,
    engineer: PriceFeatureEngineer,
    trainer: ReturnModelTrainer,
    output_csv: Path,
    limit: int | None = None,
    group_targets: Dict[str, int] | None = None,
    group_lookup: Dict[str, str] | None = None,
) -> Tuple[List[ModelPerformance], Dict[Tuple[str, str], Dict]]:
    results: List[ModelPerformance] = []
    inference_data: Dict[Tuple[str, str], Dict] = {}
    successes_total = 0
    successes_by_group: Dict[str, int] = {
        key: 0 for key in (group_targets or {})}
    ticker_list = list(tickers)
    if limit:
        ticker_iter = tqdm(total=limit, desc=f"{name} universe", unit="valid")
        iterable = ticker_list
    else:
        ticker_iter = None
        iterable = tqdm(ticker_list, desc=f"{name} universe", unit="ticker")
    for ticker in iterable:
        try:
            # data loading, preprocessing, and feature creation
            dataset = prepare_dataset(ticker, manager, preprocessor, engineer)
        except Exception as exc:
            continue
        if len(dataset) < 400:
            continue
        # preprocessing: split into train/validation/test sets
        train_df, val_df, test_df = preprocessor.split_datasets(dataset)
        if test_df["target"].nunique() < 2:
            continue
        # separate features and target
        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
        # preprocessing: scale features
        scaled_train, scaled_val, scaled_test, _ = preprocessor.scale_three(
            X_train, X_val, X_test
        )
        # training: train models with gridsearchcv on training data
        best_models = trainer.train(scaled_train, y_train)
        # refit best models on combined train+validation data
        train_val_features = pd.concat([scaled_train, scaled_val])
        train_val_labels = pd.concat([y_train, y_val])
        ticker_success = False
        for model_name, pipeline in best_models.items():
            pipeline.fit(train_val_features, train_val_labels)
            # evaluation: evaluate on test set
            metrics = trainer.evaluate(pipeline, scaled_test, y_test)
            performance = ModelPerformance(
                ticker=ticker,
                model_name=model_name,
                accuracy=float(metrics["accuracy"]),
                precision=float(metrics["precision"]),
                roc_auc=float(metrics["roc_auc"]),
                best_params=pipeline.get_params(),
            )
            results.append(performance)
            # store inference data for backtesting
            inference_data[(ticker, model_name)] = {
                "prices": dataset.loc[X_test.index, ["open", "high", "low", "close", "volume"]].copy(),
                "predictions": metrics["predictions"],
                "probabilities": metrics["probabilities"],
            }
            ticker_success = True
        if ticker_success:
            group = None
            if group_lookup:
                group = group_lookup.get(ticker, "nyse")
            if limit:
                successes_total += 1
                ticker_iter.update(1)
                ticker_iter.set_postfix(
                    successes=successes_total, refresh=False)
            if group and group_targets:
                successes_by_group[group] = successes_by_group.get(
                    group, 0) + 1
            if limit and successes_total >= limit:
                break
    if limit and ticker_iter:
        ticker_iter.close()
    summarize_results(results, output_csv)
    return results, inference_data


def _best_by_ticker(results: Iterable[ModelPerformance]) -> List[ModelPerformance]:
    best: Dict[str, ModelPerformance] = {}
    for perf in results:
        current = best.get(perf.ticker)
        if current is None or perf.accuracy > current.accuracy:
            best[perf.ticker] = perf
    return list(best.values())


# backtesting: run backtests for top models with both strategies
def _run_backtests_for_top_models(
    entries: List[ModelPerformance],
    inference_data: Dict[Tuple[str, str], Dict],
    backtester: BacktestEngine,
    reports_dir: Path,
) -> None:
    for perf in entries:
        key = (perf.ticker, perf.model_name)
        if key not in inference_data:
            continue
        # prepare prediction frame for backtesting
        dataset = build_prediction_frame(
            inference_data[key]["prices"],
            inference_data[key]["predictions"],
            inference_data[key]["probabilities"],
        )
        # backtesting: run both strategies and generate reports
        for strategy in (ProbabilitySignalStrategy, TrendFilteredStrategy):
            report_path = reports_dir / \
                f"{perf.ticker}_{perf.model_name}_{strategy.__name__}.html"
            sharpe, max_drawdown = backtester.run(
                dataset, strategy, report_path)
            perf.sharpe = sharpe
            perf.max_drawdown = max_drawdown
            perf.report_path = report_path


# backtesting: generate reports for top 10 models and create ranking csv
def _generate_top10_reports(
    large_best: List[ModelPerformance],
    inference_data: Dict[Tuple[str, str], Dict],
    backtester: BacktestEngine,
    reports_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    # select top 10 by accuracy
    top_candidates = sorted(
        large_best, key=lambda perf: perf.accuracy, reverse=True)[:10]
    rows = []
    for perf in top_candidates:
        key = (perf.ticker, perf.model_name)
        if key not in inference_data:
            continue
        # prepare prediction frame for backtesting
        dataset = build_prediction_frame(
            inference_data[key]["prices"],
            inference_data[key]["predictions"],
            inference_data[key]["probabilities"],
        )
        # backtesting: run backtest and generate report
        report_path = reports_dir / \
            f"{perf.ticker}_{perf.model_name}_top10.html"
        sharpe, max_dd = backtester.run(
            dataset, ProbabilitySignalStrategy, report_path)
        rows.append(
            {
                "ticker": perf.ticker,
                "model": perf.model_name,
                "accuracy": perf.accuracy,
                "precision": perf.precision,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "report_path": str(report_path),
            }
        )
    # create ranking dataframe sorted by sharpe ratio
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            by=["sharpe_ratio", "max_drawdown"], ascending=[False, True])
        ranking.to_csv(output_path, index=False)
    return ranking


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vidit Pokharna Project 2 Pipeline")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2021-11-12")
    parser.add_argument("--small-tickers-file", default="data/tickers.csv")
    parser.add_argument("--large-tickers-file",
                        default="data/tickers_nyse.csv")
    parser.add_argument("--nasdaq-tickers-file",
                        default="data/tickers_nasd.csv")
    parser.add_argument("--nasdaq-fraction", type=float, default=0.5,
                        help="share of large-limit pulled from NASDAQ")
    parser.add_argument("--large-limit", type=int, default=0,
                        help="optional cap on large-universe tickers")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-file", default="outputs/project2.log")
    parser.add_argument("--fill-method", default="ffill",
                        choices=["ffill", "bfill", "interpolate"])
    parser.add_argument("--scaler", default="standard",
                        choices=["standard", "minmax"])
    parser.add_argument(
        "--skip-backtests",
        action="store_true",
        help="skip Backtrader/QuantStats runs (useful for faster iterations)",
    )
    return parser.parse_args(args)


# pipeline: main orchestration function
def main(cli_args: List[str] | None = None) -> None:
    args = parse_arguments(cli_args or [])
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # initialize components
    config = MarketDataConfig(
        start_date=args.start_date, end_date=args.end_date)
    manager = MarketDataManager(config)
    preprocessor = DataPreprocessor(
        fill_method=args.fill_method, scaler=args.scaler)
    engineer = PriceFeatureEngineer()
    trainer = ReturnModelTrainer()
    backtester = BacktestEngine()

    # data loading: load ticker lists
    small_tickers = load_tickers(Path(args.small_tickers_file))
    large_tickers = load_tickers(Path(args.large_tickers_file))
    nasdaq_path = Path(
        args.nasdaq_tickers_file) if args.nasdaq_tickers_file else None
    nyse_universe = list(large_tickers)
    nasdaq_tickers = []
    if nasdaq_path and nasdaq_path.exists():
        nasdaq_tickers = load_tickers(nasdaq_path)

    # combine nyse and nasdaq universes
    nasdaq_set = set(nasdaq_tickers)
    combined_universe = list(set(nyse_universe).union(nasdaq_set))
    group_lookup = {ticker: ("nasdaq" if ticker in nasdaq_set else "nyse")
                    for ticker in combined_universe}
    large_limit = None
    group_targets: Dict[str, int] | None = None
    if args.large_limit and args.large_limit > 0:
        rng = random.Random(42)
        cache_path = config.cache_directory
        cached_candidates = [
            ticker
            for ticker in combined_universe
            if (cache_path / f"{ticker}.csv").exists()
        ]
        large_limit = args.large_limit
        nyse_target = max(0, int(large_limit * (1.0 - args.nasdaq_fraction)))
        nasdaq_target = max(0, large_limit - nyse_target)
        group_targets = {"nyse": nyse_target, "nasdaq": nasdaq_target}
        if len(cached_candidates) >= large_limit:
            rng.shuffle(cached_candidates)
            large_tickers = cached_candidates
        else:
            large_tickers = combined_universe[:]
            rng.shuffle(large_tickers)
    else:
        large_tickers = sorted(combined_universe)

    # pipeline: evaluate small and large universes (includes data loading, preprocessing, feature creation, training, evaluation)
    small_csv = output_dir / "small_universe_metrics.csv"
    large_csv = output_dir / "large_universe_metrics.csv"
    small_results, small_inference = evaluate_universe(
        "Small", small_tickers, manager, preprocessor, engineer, trainer, small_csv
    )
    large_results, large_inference = evaluate_universe(
        "Large",
        large_tickers,
        manager,
        preprocessor,
        engineer,
        trainer,
        large_csv,
        limit=large_limit,
        group_targets=group_targets,
        group_lookup=group_lookup if args.large_limit and args.large_limit > 0 else None,
    )
    combined_inference = {**small_inference, **large_inference}

    # select best models per ticker and top 2 overall
    small_best = _best_by_ticker(small_results)
    large_best = _best_by_ticker(large_results)
    top_overall = sorted(small_best + large_best,
                         key=lambda perf: perf.accuracy, reverse=True)[:2]
    ranking = pd.DataFrame()
    top_reports = []
    # backtesting: run backtests for top models and generate reports
    if not args.skip_backtests:
        _run_backtests_for_top_models(
            top_overall, combined_inference, backtester, reports_dir)
        ranking = _generate_top10_reports(
            large_best, combined_inference, backtester, reports_dir, output_dir / "top10_rankings.csv"
        )
        top_reports = [str(perf.report_path)
                       for perf in top_overall if perf.report_path]

    final_message = {
        "small_metrics": str(small_csv),
        "large_metrics": str(large_csv),
        "top_model_reports": top_reports,
        "top10_rankings": str(output_dir / "top10_rankings.csv") if not ranking.empty else "",
    }


if __name__ == "__main__":
    main(sys.argv[1:])
