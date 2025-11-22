# Interim Project 2 – ISyE 6767 Data Analysis and Machine Learning

Python Environment
------------------
- pandas 1.5.3
- numpy 1.26.1
- scikit-learn 1.3.2
- backtrader 1.9.78.123
- quantstats 0.0.77
- yfinance 0.2.66
- tqdm 4.66.1


Files
-----
- `Pokharna_Vidit_module1.py` – data acquisition (local CSV/Yahoo Finance with caching), preprocessing, feature engineering, and Backtrader `CustomCSVData`
- `Pokharna_Vidit_module2.py` – model configs (gradient boosting & random forest), GridSearchCV trainer, evaluation helpers, Backtrader strategies, QuantStats backtesting/reporting (includes frequency alias patches for QuantStats compatibility)
- `Pokharna_Vidit_project2.py` – CLI pipeline orchestrating small/large universes, model training, evaluation, top-model backtests, and ranking exports
- `data/` – provided csv references (`aapl.csv`, `002054.XSHE.csv`, `ERCOTDA_price.csv`) and ticker universes (`tickers.csv`, `tickers_nyse.csv`, `tickers_nasd.csv`)
- `outputs/` – generated metrics (`small_universe_metrics.csv`, `large_universe_metrics.csv`, `top10_rankings.csv`) plus `reports/` (QuantStats HTML)
- `cache/` – cached downloaded price data (created automatically)

Running the Pipeline
--------------------
```
python3.11 Pokharna_Vidit_project2.py \
  --small-tickers-file data/tickers.csv \
  --large-tickers-file data/tickers_nyse.csv \
  --nasdaq-tickers-file data/tickers_nasd.csv \
  --start-date 2000-01-01 \
  --end-date 2021-11-12 \
  --output-dir outputs \
  --large-limit 200 \
  --nasdaq-fraction 0.5
```

Useful flags:
- `--fill-method {ffill,bfill,interpolate}` – missing data policy (default `ffill`)
- `--scaler {standard,minmax}` – feature normalization
- `--nasdaq-tickers-file` – optional file merged with the NYSE universe before large-scale evaluation
- `--nasdaq-fraction <f>` – when `--large-limit` is set, targets `f` of the *successful* tickers to come from NASDAQ (default 0.5)
- `--large-limit <N>` – process up to `N` large-universe tickers with usable data (0 means all); if at least `N` cached CSVs exist under `cache/`, they are sampled first before falling back to live downloads
- `--skip-backtests` – skip Backtrader/QuantStats runs (useful when iterating on models only)

Outputs
-------
- `small_universe_metrics.csv` / `large_universe_metrics.csv` – per-ticker metrics (accuracy, precision, ROC-AUC, Sharpe/Max Drawdown when backtested, model hyperparameters)
- `top10_rankings.csv` – top 10 stocks (by accuracy) from the large universe including Sharpe ratio and Max Drawdown, sorted descending by Sharpe. Note: Some entries may have empty Sharpe/Max Drawdown if the strategy generated no trades.
- `reports/*.html` – QuantStats/pyfolio HTML reports:
  - Top 2 models: Reports for both `ProbabilitySignalStrategy` and `TrendFilteredStrategy` (if trades occurred)
  - Top 10 rankings: Individual reports for each of the top 10 models using `ProbabilitySignalStrategy`

Notes
-----
- Loader order: local CSVs (`data/stock_dfs`, `data`) and then Yahoo Finance via yfinance. Successful downloads are cached under `cache/`
- Each ticker is processed independently (feature frame, scaling, GridSearchCV tuning, evaluation, trading strategy backtests). Data is split 60%/20%/20% (train/validation/test); models are tuned on the training slice, refit on train+validation, and final metrics are reported on the 20% hold-out test set.
- **QuantStats Compatibility:** The code includes automatic patches for QuantStats frequency aliases (YE→A, ME→M, QE→Q) to ensure compatibility with pandas.
- **Report Generation:** If a strategy generates no trades (all predictions are 0 or no positions taken), the QuantStats report will be skipped with a warning. This is expected behavior and metrics will still be calculated.
