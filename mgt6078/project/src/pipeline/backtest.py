"""
quintile backtest utilities
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


def compute_quintile_spreads(
    frame: pd.DataFrame,
    signal_col: str,
    return_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    compute q5 minus q1 spreads by date
    """

    # guideline §6 discussion l53-l74
    if frame.empty:
        raise ValueError("frame is empty")
    logger.info("backtest evaluating %s rows across %s dates", len(frame), frame[date_col].nunique())
    assigned = (
        frame.groupby(date_col, group_keys=False)
        .apply(_assign_quintiles, signal_col=signal_col)
        .reset_index(drop=True)
    )
    avg_returns = (
        assigned.groupby([date_col, "quintile"])[return_col]
        .mean()
        .unstack("quintile")
        .rename(columns=str)
    )
    required = {"1", "5"}
    missing = required - set(avg_returns.columns)
    if missing:
        raise ValueError(f"missing quintiles {missing}")
    avg_returns["q5_q1"] = avg_returns["5"] - avg_returns["1"]
    logger.info("computed spreads for %s dates", len(avg_returns))
    return avg_returns.reset_index()


def _assign_quintiles(group: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    helper ranking within month
    """

    # guideline §6 discussion l53-l74
    ranked = group.sort_values(signal_col)
    ranked["rank"] = ranked[signal_col].rank(method="first")
    ranked["quintile"] = pd.qcut(ranked["rank"], 5, labels=["1", "2", "3", "4", "5"])
    return ranked.drop(columns=["rank"])
