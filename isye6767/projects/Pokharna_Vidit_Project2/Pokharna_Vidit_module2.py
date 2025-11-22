from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import backtrader as bt
import numpy as np
import pandas as pd
import quantstats as qs
from quantstats import stats as qs_stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline


# patch quantstats to handle invalid frequency aliases (ye->a, me->m, qe->q)
def _patch_gain_to_pain_ratio():
    original_fn = qs_stats.gain_to_pain_ratio

    def wrapper(returns, rf=0.0, resolution="M"):
        if resolution:
            upper = str(resolution).upper()
            if upper == "ME":
                resolution = "M"
            elif upper == "YE":
                resolution = "Y"
            elif upper == "QE":
                resolution = "Q"
        return original_fn(returns, rf, resolution)

    qs_stats.gain_to_pain_ratio = wrapper


def _patch_pandas_resample_methods():
    if not hasattr(pd.Series, '_original_resample'):
        pd.Series._original_resample = pd.Series.resample
    if not hasattr(pd.DataFrame, '_original_resample'):
        pd.DataFrame._original_resample = pd.DataFrame.resample

    frequency_map = {'YE': 'A', 'ME': 'M', 'QE': 'Q'}

    def patched_series_resample(self, rule, *args, **kwargs):
        rule_str = str(rule).upper()
        if rule_str in frequency_map:
            rule = frequency_map[rule_str]
        return pd.Series._original_resample(self, rule, *args, **kwargs)

    def patched_df_resample(self, rule, *args, **kwargs):
        rule_str = str(rule).upper()
        if rule_str in frequency_map:
            rule = frequency_map[rule_str]
        return pd.DataFrame._original_resample(self, rule, *args, **kwargs)

    pd.Series.resample = patched_series_resample
    pd.DataFrame.resample = patched_df_resample


def _patch_quantstats_frequencies():
    _patch_pandas_resample_methods()
    _patch_gain_to_pain_ratio()


_patch_quantstats_frequencies()


@dataclass
class ModelConfig:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, Iterable]


@dataclass
class ModelPerformance:
    ticker: str
    model_name: str
    accuracy: float
    precision: float
    roc_auc: float
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    report_path: Optional[Path] = None
    best_params: Optional[Dict] = None


# training: trains classification models with time-series cross-validation and hyperparameter tuning
class ReturnModelTrainer:

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.model_configs = self._build_configs()

    # training: build model configurations with parameter grids for grid search
    def _build_configs(self) -> List[ModelConfig]:
        gradient_boost = Pipeline(
            steps=[
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        )
        rf = Pipeline(
            steps=[
                ("clf", RandomForestClassifier(random_state=42)),
            ]
        )
        configs = [
            ModelConfig(
                name="gradient_boosting",
                pipeline=gradient_boost,
                param_grid={
                    "clf__learning_rate": [0.01, 0.05, 0.1],
                    "clf__n_estimators": [100, 200],
                    "clf__max_depth": [2, 3],
                },
            ),
            ModelConfig(
                name="random_forest",
                pipeline=rf,
                param_grid={
                    "clf__n_estimators": [100, 200],
                    "clf__max_depth": [3, 5, None],
                    "clf__min_samples_leaf": [1, 5, 10],
                },
            ),
        ]
        return configs

    # training: train models using gridsearchcv with time-series cross-validation
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Pipeline]:
        best_models: Dict[str, Pipeline] = {}
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for config in self.model_configs:
            # grid search with time-series aware cross-validation
            grid = GridSearchCV(
                estimator=config.pipeline,
                param_grid=config.param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                grid.fit(X, y)
            best_models[config.name] = grid.best_estimator_
        return best_models

    # evaluation: evaluate model performance on test set and return metrics
    @staticmethod
    def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        try:
            roc_auc = roc_auc_score(y_test, proba)
        except ValueError:
            roc_auc = float("nan")
        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc,
            "predictions": predictions,
            "probabilities": proba,
        }


class PredictionSignalData(bt.feeds.PandasData):

    lines = ("prediction", "probability")
    params = (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", "openinterest"),
        ("prediction", "prediction"),
        ("probability", "probability"),
    )


# backtesting strategy: trades based on model prediction (1=buy, 0=sell)
class ProbabilitySignalStrategy(bt.Strategy):

    params = dict(risk_fraction=0.95)

    def __init__(self):
        self.prediction = self.datas[0].prediction
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            pass

        self.order = None

    def next(self):
        pred = self.prediction[0]
        if np.isnan(pred):
            return

        pred = int(pred)

        # buy signal: prediction is 1 and no current position
        if pred == 1 and not self.position and self.order is None:
            cash = self.broker.getcash() * self.p.risk_fraction
            size = int(cash / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)
        # sell signal: prediction is 0 and have position
        elif pred == 0 and self.position and self.order is None:
            self.order = self.close()


# backtesting strategy: combines ml predictions with trend filter and trailing stop
class TrendFilteredStrategy(bt.Strategy):

    params = dict(trend_period=50, trailing_stop=0.03)

    def __init__(self):
        self.prediction = self.datas[0].prediction
        self.trend = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.trend_period)
        self.entry_price = None
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price

        self.order = None

    def next(self):
        dt = self.data.datetime.datetime(0)
        pred = self.prediction[0]
        price = self.data.close[0]
        trend_val = self.trend[0]

        if np.isnan(pred):
            return

        pred = int(pred)

        # entry: require prediction=1 and price above trend
        if not self.position and self.order is None and pred == 1 and price > trend_val:
            cash = self.broker.getcash()
            size = int(cash / price)
            if size > 0:
                self.order = self.buy(size=size)
        # exit: if prediction=0, price below trend, or trailing stop hit
        elif self.position and self.order is None:
            stop_price = self.entry_price * \
                (1 - self.p.trailing_stop) if self.entry_price else 0
            exit_condition = pred == 0 or price < trend_val or (
                stop_price and price < stop_price)
            if exit_condition:
                self.order = self.close()
                self.entry_price = None


# backtesting: runs backtests using backtrader and generates quantstats html reports
class BacktestEngine:

    def __init__(self, cash: float = 100000.0, commission: float = 0.001):
        self.cash = cash
        self.commission = commission

    def _build_cerebro(self, df: pd.DataFrame, strategy: bt.Strategy, **kwargs) -> bt.Cerebro:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(self.commission)
        feed = PredictionSignalData(dataname=df)
        cerebro.adddata(feed)
        cerebro.addstrategy(strategy, **kwargs)
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
        cerebro.addanalyzer(
            bt.analyzers.SharpeRatio_A,
            riskfreerate=0.0,
            timeframe=bt.TimeFrame.Days,
            factor=252,
            _name="sharpe",
        )
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        return cerebro

    # backtesting: run backtest and generate quantstats html report
    def run(
        self, df: pd.DataFrame, strategy: bt.Strategy, report_path: Path, **kwargs
    ) -> Tuple[float, float]:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        cerebro = self._build_cerebro(df, strategy, **kwargs)
        strat = cerebro.run()[0]
        # extract returns from backtest
        returns, _, _, _ = strat.analyzers.pyfolio.get_pf_items()
        returns = returns.fillna(0.0)
        returns.index = pd.to_datetime(returns.index)
        returns = returns.asfreq("B", method="pad")
        freq = "D"
        if returns.abs().sum() == 0:
            return np.nan, np.nan

        # generate quantstats html report
        report_generated = False
        try:
            qs.reports.html(
                returns,
                output=str(report_path),
                title=f"{strategy.__name__} Report",
                benchmark=None,
                periods_per_year=252,
                prepare_returns=True,
                match_dates=True,
                freq=freq,
            )
            report_generated = True
        except (ValueError, KeyError, AttributeError) as exc:
            try:
                qs.reports.html(
                    returns,
                    output=str(report_path),
                    title=f"{strategy.__name__} Report",
                    benchmark=None,
                    periods_per_year=252,
                    prepare_returns=True,
                    match_dates=True,
                )
                report_generated = True
            except Exception as exc2:
                pass
        # extract sharpe ratio and max drawdown metrics
        sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analyzer.get("sharperatio", np.nan)
        drawdown = strat.analyzers.drawdown.get_analysis().get("max", {})
        max_dd = drawdown.get("drawdown", np.nan)
        return sharpe_ratio, max_dd


# backtesting: prepare dataframe with ohlcv data and model predictions for backtrader
def build_prediction_frame(
    price_frame: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    df = price_frame.copy()
    # ensure all required ohlcv columns exist
    if "open" not in df:
        df["open"] = df["close"]
    if "high" not in df:
        df["high"] = df["close"]
    if "low" not in df:
        df["low"] = df["close"]
    if "volume" not in df:
        df["volume"] = 1.0
    if "openinterest" not in df:
        df["openinterest"] = 0
    # add model predictions and probabilities
    df["prediction"] = predictions
    df["probability"] = probabilities
    # set datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(price_frame.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    df.index.name = "datetime"
    return df


def summarize_results(results: List[ModelPerformance], output_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(output_csv, index=False)
    return df


__all__ = [
    "ModelPerformance",
    "ReturnModelTrainer",
    "ProbabilitySignalStrategy",
    "TrendFilteredStrategy",
    "BacktestEngine",
    "build_prediction_frame",
    "summarize_results",
]
