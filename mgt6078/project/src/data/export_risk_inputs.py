from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DataRequest, load_ff_factors
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
PANEL_PATH = DATA_PROCESSED / "phase_3_fair_value_panel" / "full_panel.csv"
MISPRICING_OLS_PATH = DATA_PROCESSED / "mispricing_ols.csv"
MISPRICING_TS_PATH = DATA_PROCESSED / "mispricing_ts.csv"
MISPRICING_LASSO_PATH = DATA_PROCESSED / "mispricing_lasso.csv"
MISPRICING_PLS_PATH = DATA_PROCESSED / "mispricing_pls.csv"
MISPRICING_XGB_PATH = DATA_PROCESSED / "mispricing_xgb.csv"
FF38_LABELS_PATH = PROJECT_ROOT / "docs" / "ff38_siccodes.csv"

RISK_INPUT_DIR = DATA_PROCESSED / "risk_inputs"
FACTORS_OUT = RISK_INPUT_DIR / "factor_returns_ff.csv"
STOCK_FF_OUT = RISK_INPUT_DIR / "stock_monthly_data_ff.csv"
STOCK_FAMA_OUT = RISK_INPUT_DIR / "stock_monthly_data_fama.csv"
INDUSTRY_RETURNS_OUT = RISK_INPUT_DIR / "industry_returns.csv"


def _format_month_end(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    return dates.dt.to_period("M").dt.to_timestamp("M")


def _format_month_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    return dates.dt.to_period("M").dt.to_timestamp("M")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return columns


def _build_stock_frames(
    panel: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["month_end"] = _format_month_end(panel["date"])
    panel = panel.sort_values(["permno", "month_end"]).reset_index(drop=True)

    base = panel[
        [
            "permno",
            "gvkey",
            "month_end",
            "ff38_industry",
            "mktcap",
            "ret_total",
        ]
    ].copy()
    base = base.rename(
        columns={
            "month_end": "date",
            "mktcap": "market_cap",
            "ret_total": "return",
        }
    )
    industry_values = base["ff38_industry"]
    industry_map = _load_industry_code_map()
    if pd.api.types.is_numeric_dtype(industry_values):
        base["industry"] = pd.to_numeric(industry_values, errors="coerce").astype("Int64")
    elif industry_map is not None:
        base["industry"] = industry_values.map(industry_map).astype("Int64")
    else:
        base["industry"] = pd.NA
    base["industry"] = base["industry"].fillna(38).astype("Int64")
    base = base.drop(columns=["ff38_industry"])
    base["permno"] = pd.to_numeric(base["permno"], errors="coerce").astype("Int64")
    base["gvkey"] = pd.to_numeric(base["gvkey"], errors="coerce").astype("Int64")

    signal_specs = [
        (MISPRICING_OLS_PATH, "mispricing_ols"),
        (MISPRICING_TS_PATH, "mispricing_ts"),
        (MISPRICING_LASSO_PATH, "mispricing_lasso"),
        (MISPRICING_PLS_PATH, "mispricing_pls"),
        (MISPRICING_XGB_PATH, "mispricing_xgb"),
    ]
    for path, column_name in signal_specs:
        signal = _load_mispricing_signal(path, column_name)
        if signal.empty:
            base[column_name] = pd.NA
        else:
            base = base.merge(signal, on=["permno", "gvkey", "date"], how="left")

    stock_ff = base[
        [
            "date",
            "permno",
            "gvkey",
            "mispricing_ols",
            "mispricing_ts",
            "mispricing_lasso",
            "mispricing_pls",
            "mispricing_xgb",
            "industry",
            "market_cap",
            "return",
        ]
    ].copy()

    stats = _compute_beta_lt(stock_ff, factor_df)

    control_map = {
        "book_market": "book_to_market",
        "st_reversal": "st_rev_1m",
        "momentum": "momentum_12_2",
        "accruals": "accruals",
        "sue": "sue",
        "gross_profitability": "gross_profitability",
        "earnings_yield": "earnings_yield",
    }

    controls = panel[["permno", "gvkey", "month_end"]].copy()
    controls = controls.rename(columns={"month_end": "date"})
    controls["permno"] = pd.to_numeric(controls["permno"], errors="coerce").astype(
        "Int64"
    )
    controls["gvkey"] = pd.to_numeric(controls["gvkey"], errors="coerce").astype(
        "Int64"
    )

    for new_col, source in control_map.items():
        if source and source in panel.columns:
            controls[new_col] = panel[source].values
        else:
            controls[new_col] = pd.NA

    controls = controls.merge(stats, on=["permno", "gvkey", "date"], how="left")

    stock_fama = stock_ff.merge(controls, on=["permno", "gvkey", "date"], how="left")
    stock_fama = stock_fama[
        [
            "date",
            "permno",
            "gvkey",
            "mispricing_ols",
            "mispricing_ts",
            "mispricing_lasso",
            "mispricing_pls",
            "mispricing_xgb",
            "industry",
            "market_cap",
            "return",
            "beta",
            "book_market",
            "st_reversal",
            "momentum",
            "lt_reversal",
            "accruals",
            "sue",
            "gross_profitability",
            "earnings_yield",
        ]
    ]

    return stock_ff, stock_fama


def _build_industry_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["month_end"] = _format_month_end(panel["date"])
    industry_map = _load_industry_code_map()
    industry_values = panel["ff38_industry"]
    if pd.api.types.is_numeric_dtype(industry_values):
        panel["industry_code"] = pd.to_numeric(industry_values, errors="coerce").astype("Int64")
    elif industry_map is not None:
        panel["industry_code"] = industry_values.map(industry_map).astype("Int64")
    else:
        raise ValueError("FF38 industry mapping required for industry returns export")
    panel["industry_code"] = panel["industry_code"].fillna(38).astype("Int64")
    returns = (
        panel.groupby(["month_end", "industry_code"])["ret_total"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "month_end": "date",
                "industry_code": "industry",
                "ret_total": "industry_return",
            }
        )
        .sort_values(["date", "industry"])
        .reset_index(drop=True)
    )
    return returns


def _build_factor_returns(start: str, end: str) -> pd.DataFrame:
    factor_cols = ["Mkt_RF", "SMB", "HML", "Mom", "ST_Rev", "LT_Rev", "CMA", "RMW"]
    custom_path = PROJECT_ROOT / "data_raw" / "factor_returns_ff.csv"

    if custom_path.exists():
        df = _load_single_factor_file(custom_path, factor_cols)
    else:
        df = _assemble_factor_components(factor_cols)

    if df is None:
        ff = load_ff_factors(DataRequest(start=start, end=end))
        ff = ff.rename(columns={"month_end": "date"})
        ff["date"] = _format_month_end(ff["date"])
        for col in factor_cols:
            if col not in ff.columns:
                ff[col] = pd.NA
        df = ff[["date"] + factor_cols].copy()

    mask = (df["date"] >= pd.to_datetime(start)) & (
        df["date"] <= pd.to_datetime(end)
    )
    return df.loc[mask].reset_index(drop=True)


def _load_single_factor_file(path: Path, factor_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    parsed = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = parsed
    if df["date"].isna().any():
        logger.warning("Dropping rows with invalid dates in %s", path)
        df = df[df["date"].notna()].copy()
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    missing = [col for col in factor_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing factor columns: {missing}")
    return df[["date"] + factor_cols]


def _assemble_factor_components(factor_cols: list[str]) -> pd.DataFrame | None:
    raw_dir = PROJECT_ROOT / "data_raw"
    ff5_path = raw_dir / "F-F_Research_Data_5_Factors_2x3.csv"
    mom_path = raw_dir / "F-F_Momentum_Factor.csv"
    st_path = raw_dir / "F-F_ST_Reversal_Factor.csv"
    lt_path = raw_dir / "F-F_LT_Reversal_Factor.csv"

    required = [ff5_path, mom_path, st_path, lt_path]
    if not all(path.exists() for path in required):
        return None

    def _load_french_file(file_path: Path) -> pd.DataFrame:
        with file_path.open("r") as handle:
            lines = handle.readlines()
        start = None
        for idx, line in enumerate(lines):
            clean = line.strip().split(",")[0]
            if clean.isdigit() and len(clean) in (5, 6):
                start = max(idx - 1, 0)
                break
        if start is None:
            raise ValueError(f"Could not detect data start in {file_path}")
        return pd.read_csv(file_path, skiprows=start)

    def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        date_col = df.columns[0]
        df["date"] = df[date_col].astype(str).str.replace(" ", "").str.slice(0, 6)
        return df

    ff5 = _clean_dates(_load_french_file(ff5_path))
    mom = _clean_dates(_load_french_file(mom_path))
    st = _clean_dates(_load_french_file(st_path))
    lt = _clean_dates(_load_french_file(lt_path))

    ff5 = ff5.rename(
        columns={"Mkt-RF": "Mkt_RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA"}
    )
    mom_cols = {col: "Mom" for col in mom.columns if col.upper() in {"MOM", "UMD"}}
    mom = mom.rename(columns=mom_cols)
    st = st.rename(columns={"ST_Rev": "ST_Rev"})
    lt = lt.rename(columns={"LT_Rev": "LT_Rev"})

    merged = ff5.merge(mom[["date", "Mom"]], on="date", how="left")
    merged = merged.merge(st[["date", "ST_Rev"]], on="date", how="left")
    merged = merged.merge(lt[["date", "LT_Rev"]], on="date", how="left")
    merged["date"] = pd.to_datetime(merged["date"], format="%Y%m", errors="coerce")
    merged["date"] = merged["date"] + pd.offsets.MonthEnd(0)
    return merged[["date"] + factor_cols]


def _compute_beta_lt(
    stock_ff: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    df = stock_ff.merge(
        factors[["date", "Mkt_RF"]], on="date", how="left", sort=False
    ).copy()
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["permno", "date"])

    df["return_adj"] = pd.to_numeric(df["return"], errors="coerce")
    df["mkt_adj"] = pd.to_numeric(df["Mkt_RF"], errors="coerce")
    if df["mkt_adj"].abs().max(skipna=True) > 1:
        df["mkt_adj"] = df["mkt_adj"] / 100.0

    def _rolling_beta(grp: pd.DataFrame) -> pd.Series:
        r = grp["return_adj"]
        m = grp["mkt_adj"]
        cov = r.rolling(window=12, min_periods=6).cov(m)
        var = m.rolling(window=12, min_periods=6).var()
        return cov / var

    def _rolling_lt(grp: pd.DataFrame) -> pd.Series:
        log_r = np.log1p(grp["return_adj"].clip(lower=-0.999999))
        return np.expm1(
            log_r.shift(12).rolling(window=36, min_periods=12).sum()
        )

    df["beta"] = (
        df.groupby("permno", group_keys=False)
        .apply(_rolling_beta)
        .reset_index(level=0, drop=True)
    )
    df["lt_reversal"] = (
        df.groupby("permno", group_keys=False)
        .apply(_rolling_lt)
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values("_order")
    return df[["permno", "gvkey", "date", "beta", "lt_reversal"]]


def _load_industry_code_map() -> dict | None:
    if not FF38_LABELS_PATH.exists():
        logger.warning("FF38 labels not found at %s", FF38_LABELS_PATH)
        return None
    mapping = pd.read_csv(FF38_LABELS_PATH)
    required = {"ff38_id", "short_name", "description"}
    if not required.issubset(mapping.columns):
        logger.warning("FF38 mapping missing required columns %s", required)
        return None
    code_map: dict[str, int] = {}
    for _, row in mapping.iterrows():
        code = int(row["ff38_id"])
        for key in ("short_name", "description"):
            label = row.get(key)
            if isinstance(label, str) and label.strip():
                code_map[label.strip()] = code
    return code_map


def _load_mispricing_signal(path: Path, column_name: str) -> pd.DataFrame:
    if not path.exists():
        alt = path.with_name("mispricing_theilsen.csv")
        if alt.exists():
            path = alt
        else:
            logger.warning(
                "%s missing at %s; %s column will be empty", column_name, path, column_name
            )
            return pd.DataFrame(columns=["permno", "gvkey", "date", column_name])

    mis = pd.read_csv(path, parse_dates=["month_end"])
    mis = mis.rename(columns={"month_end": "date"})
    mis["date"] = _format_month_end(mis["date"])
    mis["permno"] = pd.to_numeric(mis["permno"], errors="coerce").astype("Int64")
    mis["gvkey"] = pd.to_numeric(mis["gvkey"], errors="coerce").astype("Int64")
    if column_name not in mis.columns:
        rename_candidates = ["mispricing_ols", "mispricing_theilsen"]
        for candidate in rename_candidates:
            if candidate in mis.columns:
                mis = mis.rename(columns={candidate: column_name})
                break
        else:
            mis[column_name] = pd.NA
    return mis[["permno", "gvkey", "date", column_name]]


def export_risk_inputs(start: str = "1987-03-31", end: str = "2012-12-31") -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Full panel not found at {PANEL_PATH}. Run build_fair_value_panel first."
        )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RISK_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date", "datadate"])
    factor_df = _build_factor_returns(start=start, end=end)
    stock_ff, stock_fama = _build_stock_frames(panel, factor_df)

    stock_ff.to_csv(STOCK_FF_OUT, index=False)
    stock_fama.to_csv(STOCK_FAMA_OUT, index=False)
    logger.info("wrote %s and %s", STOCK_FF_OUT, STOCK_FAMA_OUT)

    industry_returns = _build_industry_returns(panel)
    industry_returns.to_csv(INDUSTRY_RETURNS_OUT, index=False)
    logger.info("wrote %s", INDUSTRY_RETURNS_OUT)

    factor_df.to_csv(FACTORS_OUT, index=False)
    logger.info("wrote %s", FACTORS_OUT)


def main() -> None:
    export_risk_inputs()


if __name__ == "__main__":
    main()
