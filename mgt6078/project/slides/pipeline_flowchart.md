# pipeline flowchart frame

## nodes
- data ingestion (compustat pit, crsp, factor files)
- preprocessing (winsorize, merges, asset scaling)
- model estimation (ols, lasso, pls, xgb)
- mispricing computation (m = (p - v) / v)
- portfolio construction (quintile sorts, backtest)
- risk adjustments (fama–macbeth, ff3/ff5/q alphas)
- robustness suite (subsamples, factor variants, frequency)
- reporting (tables, figures, paper, slides)

## arrows
- data ingestion → preprocessing → model estimation → mispricing computation
- mispricing computation → portfolio construction → risk adjustments
- portfolio construction → robustness suite → reporting
- automation layer connects preprocessing, modeling, and reporting to `automate.py --phase n` runs

## styling cues
- use results/figures/style.json color palette
- annotate guideline references on each node, e.g., guideline §5 for empirical strategy, guideline §7 for robustness
