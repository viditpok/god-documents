"""
Data loaders aligned with Bartram & Grinblatt (2018) and the MGT6078 methods spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shutil


from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data_raw"

COMPUSTAT_FEATURES: List[str] = [
    "ATQH","DVPQH","SALEQH","SEQQH","IBQH","NIQH","XIDOQH","IBADJQH","IBCOMQH",
    "ICAPTQH","TEQQH","PSTKRQH","PPENTQH","CEQQH","PSTKQH","DLTTQH","PIQH",
    "TXTQH","NOPIQH","AOQH","LTQH","DOQH","LOQH","CHEQH","ACOQH","DVQH","LCOQH","APQH",
]

FF_COLUMNS: List[str] = [
    "Mkt_RF","SMB","HML","Mom","ST_Rev","LT_Rev","CMA","RMW","RF",
]


@dataclass
class DataRequest:
    """Simple container for date range and optional ID filter."""
    start: str
    end: str
    identifiers: Optional[Iterable[int]] = None


def _ensure_unzipped(stem: str) -> Optional[Path]:
    """
    Ensure data_raw/<stem>.csv exists; if only <stem>.csv.zip is present,
    unzip it into data_raw/, skipping macOS metadata like __MACOSX and .DS_Store.
    """
    csv_path = DATA_RAW / f"{stem}.csv"
    if csv_path.exists():
        return csv_path

    zip_path = DATA_RAW / f"{stem}.csv.zip"
    if not zip_path.exists():
        return None

    logger.info("unzipping %s into %s", zip_path, DATA_RAW)
    with ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            # skip macOS metadata
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            zf.extract(member, DATA_RAW)

    # if a __MACOSX dir still slipped through, remove it
    macosx_dir = DATA_RAW / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir, ignore_errors=True)

    return csv_path if csv_path.exists() else None


def _resolve_csv_path(stems: list[str]) -> Optional[Path]:
    """
    Return the first existing CSV path for the provided list of stems.
    """
    for stem in stems:
        path = _ensure_unzipped(stem)
        if path is not None:
            return path
    return None



def _normalise_compustat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Compustat column names to the H-suffixed names in COMPUSTAT_FEATURES.

    Handles:
      - exact matches with different case (atqh -> ATQH)
      - traditional names (ATQ, AT, SALEQ, SALE, ...) -> ATQH, SALEQH, ...
    """
    upper_to_orig = {c.upper(): c for c in df.columns}
    rename: dict[str, str] = {}

    for canon in COMPUSTAT_FEATURES:
        # 1) exact match ignoring case (e.g. atqh -> ATQH)
        orig_case_ins = upper_to_orig.get(canon)
        if orig_case_ins is not None:
            if orig_case_ins != canon:
                rename[orig_case_ins] = canon
            continue

        # 2) fall back to ATQ / AT style mapping
        base = canon[:-1]  # ATQH -> ATQ, SALEQH -> SALEQ
        candidates = [base]
        if base.endswith("Q"):
            candidates.append(base[:-1])  # ATQ -> AT, SALEQ -> SALE

        for cand in candidates:
            orig = upper_to_orig.get(cand)
            if orig is not None:
                rename[orig] = canon
                break

    if rename:
        logger.info("normalising Compustat columns: %s", rename)
        df = df.rename(columns=rename)

    return df



def load_compustat_pit(
    request: DataRequest,
    wrds_credentials: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load Compustat PIT quarterly accounting data for the requested window.

    Expects a file in data_raw named:
      - compustat.csv  (or compustat.csv.zip)
    with at least 'datadate', 'gvkey', and the 28 quarterly fundamentals
    under either WRDS names (ATQ, SALEQ, ...) or H-suffixed versions
    (ATQH, SALEQH, ...) that will be harmonised to COMPUSTAT_FEATURES.
    """
    logger.info(
        "compustat request start=%s end=%s ids=%s",
        request.start,
        request.end,
        request.identifiers,
    )

    if wrds_credentials:
        raise NotImplementedError("WRDS connectivity not implemented; use compustat csv exports in data_raw")

    path = _resolve_csv_path(["compustat_quarterly", "compustat"])
    if path is None:
        raise FileNotFoundError(
            "Expected compustat_quarterly.csv or compustat.csv (.zip) in data_raw/"
        )

    logger.info("loading Compustat from %s", path)
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]

    if "DATADATE" not in df.columns:
        raise ValueError("compustat.csv must contain a 'datadate' column")

    df["DATADATE"] = pd.to_datetime(df["DATADATE"])

    mask = (df["DATADATE"] >= request.start) & (df["DATADATE"] <= request.end)
    if request.identifiers is not None and "GVKEY" in df.columns:
        mask &= df["GVKEY"].isin(list(request.identifiers))
    df = df.loc[mask].copy()

    df = _normalise_compustat_columns(df)

    # --- FORCE CREATE H-VARIABLES FROM BASE QUARTERLY ITEMS ---
    h_map = {
        "ATQ": "ATQH",
        "SALEQ": "SALEQH",
        "REVTQ": "SALEQH",  # fallback if SALEQ missing
        "IBQ": "IBQH",
        "NIQ": "NIQH",
        "CHEQ": "CHEQH",
        "DVPQ": "DVPQH",
        "DVQ": "DVQH",
        "PPENTQ": "PPENTQH",
        "CAPXY": "CAPXYH",
        "CAPSQ": "CAPSQH",
        "DLCQ": "DLCQH",
        "DLTTQ": "DLTTQH",
        "CEQQ": "CEQQH",
        "SEQQ": "SEQQH",
        "PSTKQ": "PSTKQH",
        "TXDITCQ": "TXDITCQH",
        "TXTQ": "TXTQH",
        "PIQ": "PIQH",
    }

    for base, target in h_map.items():
        if target not in df.columns:
            if base in df.columns:
                df[target] = df[base]
            else:
                df[target] = np.nan

    # === ECONOMIC DEFAULT VALUES (BG-compliant, ensures 0 missing data) ===
    default_zero = [
        "DVPQH",
        "DVQH",
        "PSTKQH",
        "PSTKRQH",
        "TXDITCQH",
        "TXTQH",
        "AOQH",
        "LCOQH",
        "ACOQH",
        "LOQH",
        "DOQH",
        "NOPIQH",
        "ICAPTQH",
        "CAPXYH",
        "CAPSQH",
    ]

    for col in default_zero:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for feat in COMPUSTAT_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
        else:
            df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
    if "ATQH" in df.columns:
        df["ATQH"] = df["ATQH"].replace(0.0, 1.0)

    if "ATQH" not in df.columns:
        raise ValueError(
            "Could not find total assets column: tried to map ATQ/AT to ATQH "
            f"but ATQH is still missing. Available columns: {', '.join(df.columns[:40])}"
        )

    rename_ids = {col: col.lower() for col in ("GVKEY", "PERMNO", "DATADATE") if col in df.columns}
    if rename_ids:
        df = df.rename(columns=rename_ids)

    id_cols = [c for c in ("gvkey", "permno", "datadate") if c in df.columns]
    feat_cols = [c for c in COMPUSTAT_FEATURES if c in df.columns]

    missing = sorted(set(COMPUSTAT_FEATURES) - set(df.columns))
    if missing:
        logger.warning("Compustat missing some expected PIT vars: %s", missing)

    return df[id_cols + feat_cols]

def load_crsp_monthly(
    request: DataRequest,
    wrds_credentials: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load CRSP monthly data from data_raw, apply standard filters and
    construct a delisting-adjusted return `ret_total`.

    Expected file: crsp.csv or crsp.csv.zip with at least:
      permno, date, ret, prc, shrout, siccd, shrcd, exchcd, [dlret]
    """
    logger.info(
        "CRSP request start=%s end=%s ids=%s",
        request.start,
        request.end,
        request.identifiers,
    )

    if wrds_credentials:
        raise NotImplementedError("WRDS connectivity not implemented; use crsp csv exports in data_raw")

    path = _resolve_csv_path(["crsp_monthly", "crsp"])
    if path is None:
        raise FileNotFoundError("Expected crsp_monthly.csv or crsp.csv (.zip) in data_raw/")

    logger.info("loading CRSP from %s", path)
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("crsp monthly file must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"] >= request.start) & (df["date"] <= request.end)
    if request.identifiers is not None:
        mask &= df["permno"].isin(list(request.identifiers))
    df = df.loc[mask].copy()

    if "shrcd" in df.columns:
        df = df[df["shrcd"].isin([10, 11])]
    if "exchcd" in df.columns:
        df = df[df["exchcd"].isin([1, 2, 3])]

    if "ret" in df.columns:
        ret = pd.to_numeric(df["ret"], errors="coerce")
    else:
        ret = pd.Series(np.nan, index=df.index)
    if ret.isna().all() and "retx" in df.columns:
        ret = pd.to_numeric(df["retx"], errors="coerce")
    if "dlret" in df.columns:
        dlret = pd.to_numeric(df["dlret"], errors="coerce").fillna(0.0)
        ret = ret.fillna(0.0)
        df["ret_total"] = (1.0 + ret) * (1.0 + dlret) - 1.0
    else:
        df["ret_total"] = ret
    return df


def load_ff_factors(request: DataRequest) -> pd.DataFrame:
    """
    Load Ken French factor file ff.csv(.zip) in the format:

      line 1–2: comments
      line 3:   blank
      line 4:   header: ,Mkt-RF,SMB,HML,RMW,CMA,RF
      line 5+:  YYYYMM, factors in percent
      footer:   text like ' Annual Factors: January-December '

    Returns monthly factors in decimals with a month_end column,
    dropping any non-YYYYMM rows.
    """
    from pandas.tseries.offsets import MonthEnd

    logger.info("FF factors request start=%s end=%s", request.start, request.end)

    path = _ensure_unzipped("ff")
    if path is None:
        raise FileNotFoundError("Expected ff.csv or ff.csv.zip in data_raw/")

    logger.info("loading FF factors from %s", path)

    df = pd.read_csv(path, skiprows=3)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "month_str"})

    df["month_str"] = df["month_str"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month_str"], format="%Y%m", errors="coerce")
    df = df[df["month"].notna()].copy()

    df["month_end"] = df["month"] + MonthEnd(0)

    if "Mkt-RF" in df.columns:
        df = df.rename(columns={"Mkt-RF": "Mkt_RF"})

    factor_candidates = ["Mkt_RF", "SMB", "HML", "Mom", "ST_Rev", "LT_Rev", "CMA", "RMW", "RF"]
    factor_cols = [c for c in factor_candidates if c in df.columns]

    for col in factor_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.abs().median(skipna=True) > 1.0:
            series = series / 100.0
        df[col] = series

    mask = (df["month_end"] >= request.start) & (df["month_end"] <= request.end)
    out = df.loc[mask, ["month_end"] + factor_cols].sort_values("month_end").reset_index(drop=True)

    logger.info(
        "loaded %d monthly FF observations with factors %s",
        len(out),
        factor_cols,
    )
    return out


def load_synthetic_compustat(
    request: DataRequest,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    logger.info("rendering synthetic Compustat sample for %s–%s", request.start, request.end)
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start=request.start, end=request.end, freq="M")
    identifiers = list(request.identifiers) if request.identifiers else list(range(1001, 1006))
    rows = []
    for firm_id in identifiers:
        assets_base = rng.uniform(500, 5000)
        for period in periods:
            scale = rng.uniform(0.8, 1.2)
            row = {
                "gvkey": firm_id,
                "datadate": period.to_timestamp("M"),
                "ATQH": assets_base * scale,
            }
            for feat in COMPUSTAT_FEATURES[1:]:
                row[feat] = rng.uniform(0.01, 0.4) * row["ATQH"]
            rows.append(row)
    return pd.DataFrame(rows)


def load_synthetic_crsp(
    request: DataRequest,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    logger.info("rendering synthetic CRSP sample for %s–%s", request.start, request.end)
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start=request.start, end=request.end, freq="M")
    identifiers = list(request.identifiers) if request.identifiers else list(range(1001, 1006))
    records = []
    for permno in identifiers:
        price_base = rng.uniform(15, 75)
        for period in periods:
            ret = rng.normal(0.01, 0.05)
            price = max(5.0, price_base * (1 + ret))
            records.append(
                {
                    "permno": permno,
                    "date": period.to_timestamp("M"),
                    "ret": ret,
                    "ret_total": ret,
                    "shrout": rng.uniform(5, 200),
                    "prc": price,
                    "siccd": rng.integers(1000, 5900),
                    "shrcd": 10,
                    "exchcd": 1,
                }
            )
    return pd.DataFrame(records)


def export_synthetic_samples(output_dir: Path, request: DataRequest) -> None:
    logger.info("exporting synthetic samples to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comp = load_synthetic_compustat(request, seed=42)
    crsp = load_synthetic_crsp(request, seed=42)
    comp.to_csv(output_dir / "sample_compustat.csv", index=False)
    crsp.to_csv(output_dir / "sample_crsp.csv", index=False)
