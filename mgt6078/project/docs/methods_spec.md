# methods spec

## objective
- replicate bartram & grinblatt (2018) peer-implied fair value mispricing signal across march 1987–december 2012 using compustat pit and crsp monthly data
- extend baseline by integrating lasso, pls, and xgboost regressors while maintaining guideline §5 strategy requirements l37-l52

## data inputs
- compustat pit 28-variable panel (appendix b p.145) with winsorized accounting ratios scaled by assets
- crsp monthly returns, shares outstanding, and prices filtered for nyse/amex/nasdaq common stocks (share codes 10/11) with price ≥ 5 per bartram & grinblatt p.129
- factor files: kenneth french ff3, ff5, q factors; momentum, reversal, cma, rmw per guideline §7 robustness l60-l74

## preprocessing
- align accounting data to month-end observation lag using most recent 10-k/10-q as of t following bartram & grinblatt p.130
- winsorize top/bottom 5% of each variable relative to assets via `winsorize_relative_to_assets` to control outliers
- merge with crsp identifiers; enforce sic exclusion for financials (60-69) and handle delistings as in bartram & grinblatt p.129 footnote

## modeling
- baseline ols: cross-sectional regression of market cap on accounting features each month, extracting residual implied mispricing `m = (p - v) / v` (# bartram & grinblatt eq (1))
- ml extensions: lasso (coordinate descent with cross-validation), pls (latent components capturing covariance), xgboost (tree boosting with monotonic constraints disabled) trained on same feature set for peer-implied value predictions
- model evaluation relies on monthly rolling fit using data available at time t only

## backtesting
- compute mispricing quintiles monthly, form equal-weight portfolios, and evaluate t+1 returns; capture q5-q1 spread and sharpe ratios to mirror table 2 in bartram & grinblatt
- store intermediate artifacts `results/tables` and `results/figures` to ensure restartable pipeline per guideline §6 discussion l53-l74

## risk adjustments
- run fama-macbeth cross-sectional regressions with controls (beta, size, book/market, momentum, accruals, profitability) referencing guideline §7 l60-l74 requirements
- estimate alphas under ff3, ff5, and q-factor models using `linearmodels` time-series regressions

## robustness
- subsample analyses: pre/post 2008 gfc, post-2020 covid onset, size buckets
- alternative factor sets: ff3, ff5, q, and an industry-adjusted benchmark
- frequency tests: monthly baseline with optional quarterly roll-up

## automation
- orchestrate phases via `python automate.py --phase n` commands, logging to `logs/run.log` and writing checkpoints in `data_processed/phase_n/`

## deliverables
- 15-page report following guideline section order with citations to bartram & grinblatt methodology
- results tables/figures and 12-minute presentation deck summarizing what's new versus original paper
