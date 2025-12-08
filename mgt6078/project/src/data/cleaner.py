"""
data cleaning helpers honoring guideline §6 discussion of results
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


def winsorize_relative_to_assets(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    asset_col: str = "ATQH",
    quantile: float = 0.05,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Winsorize each accounting variable relative to assets, following
    Bartram & Grinblatt (2018, appendix B):

      ratio = X / assets
      clip ratio to [q, 1-q] within each cross-section (e.g. month)
      X_w = clipped_ratio * assets

    Parameters
    ----------
    df : DataFrame with gvkey, datadate, asset_col, and feature columns.
    feature_cols : variables to winsorize; if None, all non-ID columns.
    asset_col : name of the total assets column (ATQH in this project).
    quantile : lower/upper quantile for clipping (e.g. 0.05).
    group_col : optional column name for cross-sectional grouping
                (e.g. "datadate" for month-by-month winsorization).
    """
    if asset_col not in df.columns:
        raise ValueError(f"asset column {asset_col} missing from frame")

    features = list(feature_cols) if feature_cols else [
        col for col in df.columns if col not in {"gvkey", "datadate"}
    ]
    features = [c for c in features if c != asset_col]

    cleaned = df.copy()

    logger.info(
        "winsorizing %d features relative to %s (group_col=%s)",
        len(features),
        asset_col,
        group_col,
    )

    if group_col is None:
        groups = [(None, cleaned)]
    else:
        groups = cleaned.groupby(group_col, sort=False)

    for key, group in groups:
        assets = group[asset_col].replace(0, np.nan).astype("float64")

        for col in features:
            series = group[col]
            if not np.issubdtype(series.dtype, np.number):
                continue

            ratio = series.astype("float64") / assets
            lower = ratio.quantile(quantile)
            upper = ratio.quantile(1 - quantile)
            clipped = ratio.clip(lower=lower, upper=upper)

            cleaned.loc[group.index, col] = clipped * assets

    return cleaned


def add_winsorized_suffix(df: pd.DataFrame, suffix: str = "_w") -> pd.DataFrame:
    """
    Add suffix to winsorized feature columns, but keep ID/time columns unchanged.
    """
    logger.info("adding suffix %s to winsorized columns", suffix)
    renamed = df.copy()
    id_cols = {"gvkey", "datadate", "date"}  # treat 'date' as an ID too
    rename_map = {col: f"{col}{suffix}" for col in df.columns if col not in id_cols}
    return renamed.rename(columns=rename_map)

