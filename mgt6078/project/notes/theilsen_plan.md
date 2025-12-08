# theil–sen robustness plan

## sampling
- use monthly cross-sections from march 1987 through december 2012 consistent with bartram & grinblatt (2018) sample period (p.129)
- filter to nyse/amex/nasdaq common stocks (share codes 10/11), price ≥ 5, sic not in 60–69, mirroring baseline filters
- ensure accounting variables align with most recent filings available at t, aggregating quarterly income/cash flow items over prior four quarters (bartram & grinblatt p.130)

## estimation steps
1. construct design matrix `x_t` with 28 compustat pit items, augmented with intercept
2. for each firm j in month t, compute theil–sen slope median across pairwise combinations:
   - sample pairwise slopes of `(v_k - v_l) / (x_k - x_l)` for each feature (excluding zeros) to approximate robust regression weights
   - capture intercept via median of `v_j - x_j * beta_ts`
3. derive peer-implied value `p_j,t` as `x_j beta_ts + intercept_ts`; mispricing `m_j,t = (p_j,t - v_j,t) / v_j,t` (# bartram & grinblatt robust alternative, section 4.1)
4. log coefficient dispersion and compare against ols coefficients to monitor robustness

## implementation notes
- leverage `sklearn.linear_model.TheilSenRegressor` with custom loss tuned for dense features; fallback to manual median-of-slopes if package constraints arise
- maintain consistent feature scaling: divide each accounting variable by total assets before fitting, then rescale predictions to market capitalization level
- store monthly coefficient snapshots in `results/tables/theilsen_coefficients.csv` for reproducibility
- integrate with phase 4 robustness suite to benchmark performance relative to ols, lasso, pls, xgboost
