from pathlib import Path

import pandas as pd
import yaml

CFG = yaml.safe_load(open("config/project.yaml"))
RAW = Path(CFG["data"]["out_raw"])
INT = Path(CFG["data"]["out_interim"])
PRO = Path(CFG["data"]["out_processed"])
INT.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)

TARGETS = ["HML", "SMB", "UMD"]  # predict next-month returns of these factors
REQUIRED_FF_COLS = set(TARGETS) | {"Mkt_RF", "RF"}


def main() -> None:
    # --- Load ---
    fred = pd.read_parquet(RAW / "fred_monthly.parquet").sort_index()
    ff = pd.read_parquet(RAW / "ff/ff_monthly.parquet").sort_index()

    # --- Load NLP features (lag per config to avoid lookahead) ---
    from pathlib import Path as _P

    nlp_path = INT / "nlp_regime_scores.parquet"
    if _P(nlp_path).exists():
        nlp = pd.read_parquet(nlp_path).sort_index()
        lag_nlp = int(CFG.get("lags", {}).get("nlp_months", 1))
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
    ff = ff.loc[start:end]

    # --- Sanity check FF columns ---
    missing = REQUIRED_FF_COLS - set(ff.columns)
    if missing:
        raise ValueError(f"FF dataset missing columns: {missing}. Re-run fetch_ff.py.")

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

    # --- Targets y(t+1): next-month factor returns ---
    Y = ff[TARGETS].shift(-1)

    # --- Aux benchmark columns (at t) ---
    aux = ff[["Mkt_RF", "RF"]]

    # --- Align on common index explicitly ---
    common = X.index.intersection(Y.index).intersection(aux.index)
    X, Y, aux = X.loc[common], Y.loc[common], aux.loc[common]

    # --- Build final panel and clean ---
    data = pd.concat([X, Y, aux], axis=1)

    # Drop rows missing targets while allowing optional NLP gaps.
    target_cols = [*TARGETS, "Mkt_RF", "RF"]
    data = data.dropna(subset=target_cols).sort_index()

    if data.empty:
        raise ValueError(
            "Panel is empty after alignment and NA drop. "
            "Check date ranges and that FRED/FF files share overlapping months."
        )

    # --- Save ---
    out_path = PRO / "panel.parquet"
    data.to_parquet(out_path)
    print(f"Saved panel: {data.shape} -> {out_path}")


if __name__ == "__main__":
    main()
