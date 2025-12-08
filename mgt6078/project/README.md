# Machine Learning Peer-Implied Fair Value

Machine Learning Peer-Implied Fair Value: Replicating and Extending Bartram & Grinblatt (2018) commits to reproducing the peer-implied fair value OLS results over 1987–2012 with Compustat PIT and CRSP data, applying Lasso, PLS, and XGBoost extensions, satisfying the MGT6078 project guideline’s eight-section report, robustness menu, 12-minute presentation, and end-to-end reproducibility expectations.


## Code Running Structure

Run these commands from the project root, in order:

1. `python -m src.pipeline.run_pipeline --start 1987-03-31 --end 2012-12-31`
2. `python -m src.pipeline.run_ols_baseline --min-obs 0`
3. `python -m src.pipeline.run_theilsen_peer --min-obs 0`
4. `python -m src.pipeline.run_ml_models --min-obs 0`
5. `python -m src.data.export_risk_inputs`

### What each step does
- `run_pipeline` rebuilds the Compustat–CRSP panel, merges in all Table 3 controls (beta, size, book/market, 12–2 momentum, short/long reversal, accruals, profitability, earnings yield, SUE, FF38 industry), winsorizes the 28 accounting features, and saves `full_panel.csv` plus `design_matrix.csv` under `data_processed/phase_3_fair_value_panel/`.
- `run_ols_baseline --min-obs 0` fits the peer-implied OLS fair-value model for every month, producing `data_processed/mispricing_ols.csv` with the full 1987–2012 coverage.
- `run_theilsen_peer --min-obs 0` applies the robust Theil–Sen variant to the same design matrix so `data_processed/mispricing_ts.csv` is filled for every firm-month.
- `run_ml_models --min-obs 0` fits the Lasso, PLS, and XGBoost peer fair-value regressions (same feature set as OLS) and saves `mispricing_lasso.csv`, `mispricing_pls.csv`, and `mispricing_xgb.csv` under `data_processed/`.
- `python -m src.data.export_risk_inputs` packages everything Yi needs: it writes `data_processed/risk_inputs/stock_monthly_data_ff.csv`, `stock_monthly_data_fama.csv`, `industry_returns.csv`, and `factor_returns_ff.csv`. The FF file currently carries FF5 (Mkt_RF, SMB, HML, RMW, CMA, RF) but the stock files retain `Mom`, `ST_Rev`, and `LT_Rev` placeholders so we can bolt on six/eight-factor regressions later. Each mispricing model should append its signal to both stock CSVs using `permno, gvkey, date, mispricing_<model>` to keep formats consistent for the risk-adjusted scripts.

## Repository Layout
```
data_raw/
data_processed/
src/data/
src/models/
src/pipeline/
results/
  tables/
  figures/
paper/
docs/
slides/
```
