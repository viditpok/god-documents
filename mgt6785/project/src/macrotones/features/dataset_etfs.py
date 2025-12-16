from pathlib import Path

import pandas as pd
import yaml

CFG = yaml.safe_load(open("config/project.yaml"))
RAW = Path(CFG["data"]["out_raw"])
INT = Path(CFG["data"]["out_interim"])
PRO = Path(CFG["data"]["out_processed"])
INT.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)

# ETF Targets (Long-Only)
TARGETS = ["Value", "Size", "Momentum", "Quality", "LowVol", "Growth"]
MARKET = "Market"
REQUIRED_COLS = set(TARGETS) | {MARKET}


def main() -> None:
    # --- Load ---
    fred = pd.read_parquet(RAW / "fred_monthly.parquet").sort_index()
    etf_path = RAW / "etfs/etfs_monthly.parquet"

    if not etf_path.exists():
        raise FileNotFoundError(
            f"ETF data not found at {etf_path}. Run fetch_etfs.py first."
        )

    etfs = pd.read_parquet(etf_path).sort_index()

    # --- Load NLP features (lag per config to avoid lookahead) ---
    from pathlib import Path as _P

    nlp_path = INT / "nlp_regime_scores.parquet"
    if _P(nlp_path).exists():
        nlp = pd.read_parquet(nlp_path).sort_index()
        lag_nlp = int(CFG.get("lags", {}).get("nlp_months", 1))
        # ... (Same logic as dataset.py for NLP) ...
        preferred_cols = [c for c in nlp.columns if c.endswith("_filled")]
        retain_cols = sorted(
            set(preferred_cols) | {"pos_minus_neg", "pos_minus_neg_filled", "cov_docs"}
        )
        existing_cols = [c for c in retain_cols if c in nlp.columns]
        if not existing_cols:
            existing_cols = list(nlp.columns)
        nlp = nlp[existing_cols].shift(lag_nlp)
        fred = fred.join(nlp, how="left")
        if "cov_docs" in fred.columns:
            fred["nlp_cov_docs"] = fred["cov_docs"]

    # --- Window to project dates ---
    start = pd.to_datetime(CFG["project"]["start"])
    end = pd.to_datetime(CFG["project"]["end"])
    fred = fred.loc[start:end]
    etfs = etfs.loc[start:end]

    # --- Sanity check columns ---
    missing = REQUIRED_COLS - set(etfs.columns)
    if missing:
        raise ValueError(
            f"ETF dataset missing columns: {missing}. Re-run fetch_etfs.py."
        )

    # --- Technical Features (calculated on raw unlagged ETF data) ---
    from macrotones.features.technicals import compute_realized_vol, compute_rsi
    
    # Reconstruct pseudo-prices for RSI calculation (since we only have returns)
    # Start at 100.0
    prices = (1 + etfs).cumprod() * 100.0
    
    tech_dfs = []
    # Calculate for factors + market
    for col in [*TARGETS, MARKET]:
        if col not in etfs.columns:
            continue

        # RSI needs Price history
        rsi = compute_rsi(prices[col], window=14).rename(f"{col}_RSI")
        
        # Volatility needs Returns history
        vol = compute_realized_vol(etfs[col], window=12).rename(f"{col}_Vol")
        
        # Shift technicals by 1 month to use t info for t+1 prediction
        tech_dfs.append(rsi.shift(1))
        tech_dfs.append(vol.shift(1))

    if tech_dfs:
        tech_features = pd.concat(tech_dfs, axis=1)
    else:
        tech_features = pd.DataFrame(index=etfs.index)

    # --- Features X(t): apply global and per-series release lags ---
    lag_cfg = CFG.get("lags", {})
    global_fred_lag = int(lag_cfg.get("fred_months", 1))
    fred_lagged = fred.shift(global_fred_lag).copy()
    for col, lag_value in lag_cfg.items():
        if col in fred.columns and isinstance(lag_value, int | float):
            fred_lagged[col] = fred[col].shift(global_fred_lag + int(lag_value))

    # keep TERM_SPREAD computed after lagging yields
    if {"DGS10", "DGS1"} <= set(fred_lagged.columns):
        fred_lagged["TERM_SPREAD"] = fred_lagged["DGS10"] - fred_lagged["DGS1"]

    # NLP parquet already lagged above; use lagged macro as X
    X = fred_lagged.copy()
    if not tech_features.empty:
        X = X.join(tech_features, how="left")

    # --- Targets y(t+1): next-month factor returns ---
    Y = etfs[TARGETS].shift(-1)

    # --- Aux benchmark columns (at t) ---
    # Treat Market as Mkt_RF equivalent (proxy) and use RF=0
    # (since distinct RF not in ETFs usually,
    # unless we merge from FF. Let's merge RF from FF if possible,
    # or assume 0 for ETF absolute returns context).
    # Better: Load FF just for RF.
    try:
        ff = pd.read_parquet(RAW / "ff/ff_monthly.parquet").sort_index()
        rf_series = ff["RF"].reindex(etfs.index).fillna(0.0)
    except Exception:
        rf_series = pd.Series(0.0, index=etfs.index, name="RF")

    aux = pd.DataFrame({
        "Mkt_RF": etfs[MARKET],  # Using raw market return as proxy
        "RF": rf_series,
    })

    # --- Align on common index explicitly ---
    common = X.index.intersection(Y.index).intersection(aux.index)
    X, Y, aux = X.loc[common], Y.loc[common], aux.loc[common]

    # --- Build final panel and clean ---
    data = pd.concat([X, Y, aux], axis=1)

    # Drop rows missing targets
    target_cols = [*TARGETS, "Mkt_RF"]
    data = data.dropna(subset=target_cols).sort_index()

    if data.empty:
        raise ValueError("Panel is empty after alignment. Check date ranges.")

    # --- Save ---
    # SAVE AS panel_etfs.parquet so we don't overwrite the original (yet)
    out_path = PRO / "panel_etfs.parquet"
    data.to_parquet(out_path)
    print(f"Saved ETF panel: {data.shape} -> {out_path}")


if __name__ == "__main__":
    main()
