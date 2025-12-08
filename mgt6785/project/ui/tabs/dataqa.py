from __future__ import annotations

import pandas as pd
import streamlit as st


def _qa_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.isna().mean().to_frame("Missing%")
    stats["Missing%"] = stats["Missing%"].mul(100)
    stats["Last Refresh"] = pd.to_datetime(df.index.max())
    today = pd.Timestamp.today().normalize()
    stats["Vintage Lag (days)"] = (today - stats["Last Refresh"]).dt.days
    rolling_std = df.std(ddof=0).replace(0.0, pd.NA)
    outliers = (df - df.mean()).abs() > (3 * rolling_std.fillna(0))
    stats["Outlier%"] = outliers.sum() / len(df) * 100

    def _quality_icon(pct: float) -> str:
        if pct == 0:
            return "🟢"
        if pct <= 10:
            return "🟡"
        return "🔴"

    stats["QA Signal"] = stats["Missing%"].apply(_quality_icon)
    return stats.round({"Missing%": 2, "Outlier%": 2})


def render(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No panel data available. Run `make quickstart` to populate parquet.")
        return
    stats = _qa_summary(df)

    def _highlight(row: pd.Series) -> list[str]:
        color = "background-color: rgba(248,113,113,0.25)" if row["Outlier%"] > 1 else ""
        return [color] * len(row)

    styled = stats.style.apply(_highlight, axis=1)
    st.dataframe(styled, width="stretch")


__all__ = ["render", "_qa_summary"]
