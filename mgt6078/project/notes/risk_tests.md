# risk testing specification

## objectives
- quantify mispricing signal performance under cross-sectional and time-series risk adjustments mandated by guideline §7 robustness checks l60-l74
- benchmark ols, lasso, pls, xgboost models against bartram & grinblatt (2018) alphas (table 3 and table 5)

## fama–macbeth regressions
- monthly cross-sectional regression: `r_{j,t+1} = a_t + b_t m_{j,t} + controls_t + ε_{j,t+1}`
- controls:
  - market beta (12-month rolling from crsp)
  - log market cap and book-to-market (compustat)
  - momentum (12-2 month return)
  - short-term reversal (1 month)
  - accruals, gross profitability, earnings yield, suprise (appendix b notes)
  - industry fixed effects (kenneth french 38 industries)
- output: average coefficients with newey-west adjusted t-stats; table stored in `results/tables/fama_macbeth_*.csv`

## factor model alphas
- run time-series regressions of quintile portfolio excess returns on:
  - ff3 (mkt_rf, smb, hml)
  - ff5 (add rmw, cma)
  - q-factor (mkt_rf, me, ia, roe)
  - momentum and reversal add-ons per guideline §7 l60-l74
- compute intercepts (alphas), standard errors, and annualized sharpe ratios
- compare across models to document persistence of mispricing spread

## reporting
- align tables with guideline structure section 6 (discussion of results) and section 7 (robustness checks)
- include citations: # bartram & grinblatt table 3 for fama–macbeth design; # bartram & grinblatt table 5 for factor regressions
- generate companion markdown summary in `paper/03_results.md` during phase 5
