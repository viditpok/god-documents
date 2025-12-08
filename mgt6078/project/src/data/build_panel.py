from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from src.data.loader import (
    DataRequest,
    load_compustat_pit,
    load_crsp_monthly,
    load_ff_factors,
    COMPUSTAT_FEATURES,
)
from src.data.cleaner import winsorize_relative_to_assets, add_winsorized_suffix
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
FF38_MAP_PATH = PROJECT_ROOT / "docs" / "ff38_siccodes.csv"


def _load_ff38_mapping() -> pd.DataFrame:
    if not FF38_MAP_PATH.exists():
        logger.warning("FF38 mapping file missing at %s", FF38_MAP_PATH)
        return pd.DataFrame()
    mapping = pd.read_csv(FF38_MAP_PATH)
    mapping["sic_start"] = pd.to_numeric(mapping["sic_start"], errors="coerce")
    mapping["sic_end"] = pd.to_numeric(mapping["sic_end"], errors="coerce")
    mapping["ff38_id"] = pd.to_numeric(mapping["ff38_id"], errors="coerce").astype("Int64")
    return mapping


def _assign_ff38_industry(sic_series: pd.Series, mapping: pd.DataFrame) -> pd.Series:
    if mapping.empty:
        return pd.Series(pd.NA, index=sic_series.index, dtype="Int64")

    labels = pd.Series(pd.NA, index=sic_series.index, dtype="Int64")
    sic_numeric = pd.to_numeric(sic_series, errors="coerce")

    for _, row in mapping.iterrows():
        start = row["sic_start"]
        end = row["sic_end"]
        if pd.notna(start) and pd.notna(end):
            mask = sic_numeric.between(start, end, inclusive="both")
            labels.loc[mask] = row["ff38_id"]
    labels = labels.fillna(38)
    return labels


def _compute_compustat_controls(comp: pd.DataFrame) -> pd.DataFrame:
    ctrl = comp.sort_values(["gvkey", "datadate"]).copy()
    required = ["ACOQH", "LCOQH", "CHEQH", "SALEQH", "ATQH", "IBQH"]
    for col in required:
        if col not in ctrl.columns:
            ctrl[col] = np.nan

    group = ctrl.groupby("gvkey", group_keys=False)
    delta_aco = group["ACOQH"].diff()
    delta_lco = group["LCOQH"].diff()
    delta_che = group["CHEQH"].diff()

    accruals = ((delta_aco - delta_lco) - delta_che) / ctrl["ATQH"]
    gross_prof = ctrl["SALEQH"] / ctrl["ATQH"]

    earnings = ctrl["IBQH"]
    earnings_lag4 = group["IBQH"].shift(4)
    delta_earn = earnings - earnings_lag4
    rolling_std = (
        group["IBQH"]
        .rolling(window=8, min_periods=4)
        .std()
        .reset_index(level=0, drop=True)
    )
    sue = delta_earn / rolling_std

    ctrl["accruals"] = accruals.fillna(0.0)
    ctrl["gross_profitability"] = gross_prof.fillna(0.0)
    ctrl["sue"] = sue

    return ctrl[["gvkey", "datadate", "accruals", "gross_profitability", "sue"]]


def _rolling_beta(group: pd.Series, mkt: pd.Series) -> pd.Series:
    cov = group.rolling(window=12, min_periods=12).cov(mkt)
    var = mkt.rolling(window=12, min_periods=12).var()
    return cov / var


def _compute_momentum(log_returns: pd.Series) -> pd.Series:
    shifted = log_returns.shift(2)
    window = shifted.rolling(window=11, min_periods=11).sum()
    return np.expm1(window)


def build_fair_value_panel(
    start: str = "1987-03-31",
    end: str = "2012-12-31",
    lag_months: int = 3,          # <— define lag_months here
) -> None:
    """
    End-to-end fair value panel construction (firm–month):

    - load Compustat PIT (quarterly) + CRSP monthly
    - as-of merge with an observation lag of `lag_months`
    - drop financials (SIC 60–69) and price < 5
    - winsorize accounting vars relative to assets by month
    - compute market cap and t+1 returns
    - save full panel + design matrix under data_processed/phase_3_fair_value_panel
    """

    out_dir = DATA_PROCESSED / "phase_3_fair_value_panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw Compustat PIT and CRSP monthly data
    req = DataRequest(start=start, end=end)
    comp = load_compustat_pit(req)
    crsp = load_crsp_monthly(req)
    ff = load_ff_factors(req)
    ff38_map = _load_ff38_mapping()

    logger.info("compustat shape %s, crsp shape %s", comp.shape, crsp.shape)

    comp = comp.copy()
    crsp = crsp.copy()
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    crsp["date"] = pd.to_datetime(crsp["date"])

    # 2. Common firm_id + observation date for as-of merge
    if "permno" in comp.columns:
        comp["firm_id"] = comp["permno"]
    else:
        comp["firm_id"] = comp["gvkey"]

    crsp["firm_id"] = crsp["permno"]
    crsp["obs_date"] = crsp["date"] - DateOffset(months=lag_months)

    # IMPORTANT: sort primarily by the as-of key so merge_asof is happy
    comp_sorted = comp.sort_values(["datadate", "firm_id"])
    comp_controls = _compute_compustat_controls(comp_sorted)
    crsp_sorted = crsp.sort_values(["obs_date", "firm_id"])

    # 3. As-of merge: latest accounting info available by obs_date
    panel = pd.merge_asof(
        crsp_sorted,
        comp_sorted,
        left_on="obs_date",
        right_on="datadate",
        by="firm_id",
        direction="backward",
        suffixes=("", "_comp"),
    )

    panel = panel.drop(columns=["obs_date", "firm_id"])

    if "ATQH" not in panel.columns:
        raise ValueError(
            "ATQH (total assets) missing after as-of merge. "
            "Check Compustat loader and column mappings."
        )

    panel = panel[panel["ATQH"].notna()].copy()

    # 4. Sample filters and market cap
    panel = panel[panel["prc"].abs() >= 5]

    if "siccd" in panel.columns:
        sic = pd.to_numeric(panel["siccd"], errors="coerce")
        panel = panel[(sic < 6000) | (sic > 6999) | sic.isna()]

    logger.info("panel shape %s after filters", panel.shape)

    panel["mktcap"] = panel["prc"].abs() * panel["shrout"] * 1000.0
    panel["log_mktcap"] = np.where(panel["mktcap"] > 0, np.log(panel["mktcap"]), np.nan)

    ff = ff.rename(columns={"month_end": "date"})
    panel = panel.merge(ff, on="date", how="left")
    panel = panel.merge(comp_controls, on=["gvkey", "datadate"], how="left")
    panel = panel.sort_values(["permno", "date"])

    ret_col = "ret_total" if "ret_total" in panel.columns else "ret"
    panel[ret_col] = pd.to_numeric(panel[ret_col], errors="coerce")
    rf = panel["RF"] if "RF" in panel.columns else 0.0
    panel["ret_excess"] = panel[ret_col] - rf

    if "Mkt_RF" in panel.columns:
        panel["beta_12m"] = panel.groupby("permno", group_keys=False).apply(
            lambda grp: _rolling_beta(grp["ret_excess"], grp["Mkt_RF"])
        )
    else:
        panel["beta_12m"] = np.nan

    log_returns = np.log1p(panel[ret_col].clip(lower=-0.999999))
    panel["log_returns"] = log_returns
    panel["momentum_12_2"] = (
        panel.groupby("permno")["log_returns"]
        .transform(_compute_momentum)
        .replace([np.inf, -np.inf], np.nan)
    )
    panel["st_rev_1m"] = panel.groupby("permno")[ret_col].shift(1)
    panel = panel.drop(columns=["log_returns"])

    book_equity = None
    for col in ("CEQQH", "SEQQH", "TEQQH"):
        if col in panel.columns:
            if book_equity is None:
                book_equity = panel[col].copy()
            else:
                book_equity = book_equity.fillna(panel[col])
    if book_equity is None:
        book_equity = pd.Series(np.nan, index=panel.index)

    if "IBQH" not in panel.columns:
        panel["IBQH"] = np.nan
    panel["book_to_market"] = np.where(
        panel["mktcap"] > 0,
        book_equity / panel["mktcap"],
        np.nan,
    )
    panel["earnings_yield"] = panel["IBQH"] / panel["mktcap"]
    sic_series = panel["siccd"] if "siccd" in panel.columns else pd.Series(pd.NA, index=panel.index)
    panel["ff38_industry"] = _assign_ff38_industry(sic_series, ff38_map)

    # 5. Winsorize accounting vars relative to assets by month
    acct_cols = [c for c in COMPUSTAT_FEATURES if c in panel.columns]
    if not acct_cols:
        raise ValueError("No COMPUSTAT_FEATURES found in merged panel.")

    acct_frame = panel[["gvkey", "date"] + acct_cols].copy()
    acct_w = winsorize_relative_to_assets(
        acct_frame,
        feature_cols=acct_cols,
        asset_col="ATQH",
        quantile=0.05,
        group_col="date",  # cross-sectional winsorization each month
    )
    acct_w = add_winsorized_suffix(acct_w)

    panel = panel.merge(acct_w, on=["gvkey", "date"], how="left")

    # 6. Next-month returns (delisting-adjusted if available)
    ret_col = "ret_total" if "ret_total" in panel.columns else "ret"
    panel["ret_t1"] = panel.groupby("permno")[ret_col].shift(-1)

    # 7. Save full panel and design matrix
    feature_cols = [c for c in panel.columns if c.endswith("_w")]

    control_candidates = [
        "beta_12m",
        "log_mktcap",
        "book_to_market",
        "momentum_12_2",
        "st_rev_1m",
        "accruals",
        "gross_profitability",
        "earnings_yield",
        "sue",
        "ff38_industry",
    ]
    control_cols = [c for c in control_candidates if c in panel.columns]

    design = panel[["permno", "gvkey", "date", "mktcap"] + feature_cols + control_cols].copy()
    design = design.rename(columns={"date": "month_end"})

    out_full = out_dir / "full_panel.csv"
    out_design = out_dir / "design_matrix.csv"

    panel.to_csv(out_full, index=False)
    design.to_csv(out_design, index=False)

    logger.info("saved full panel to %s", out_full)
    logger.info("saved design matrix to %s with %d features", out_design, len(feature_cols))
