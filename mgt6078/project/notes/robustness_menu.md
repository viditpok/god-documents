# robustness menu

## subsamples
- pre-gfc: march 1987–december 2007
- post-gfc: january 2008–december 2012
- covid-era extension: january 2020 onward (if data available), following guideline §7 robustness checks l60-l74 emphasis on crisis windows
- size buckets: nyse breakpoints for small, mid, large caps

## factor alternatives
- ff3 baseline (mkt_rf, smb, hml)
- ff5 expanded (add rmw, cma)
- hou-xue-zhang q factors (mkt_rf, me, ia, roe)
- industry-adjusted returns using kenneth french 38-industry portfolios

## frequency variations
- monthly baseline portfolios
- quarterly rebalanced strategy using quarter-end accounting aggregation

## diagnostics
- compare mispricing spreads, alphas, and sharpe ratios across subsamples/factors
- store outputs in `results/tables/robustness_*.csv` and `results/figures/robustness_*.png`
- log key metrics per run to comply with guideline §7 l60-l74 documentation requirements
