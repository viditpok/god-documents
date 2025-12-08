from __future__ import annotations

import pandas as pd

try:  # optional dependency guard
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None


def generate_llm_summary(
    ic_df: pd.DataFrame | None,
    attribution_df: pd.DataFrame | None,
) -> str:
    """
    Offline-friendly narrative summary. This can be replaced with an
    OpenAI call when API credentials are configured.
    """

    ic_df = ic_df.copy() if ic_df is not None else pd.DataFrame()
    attribution_df = (
        attribution_df.copy() if attribution_df is not None else pd.DataFrame()
    )
    if ic_df.empty:
        return "IC summary unavailable; please run diagnostics first."

    avg_ic = ic_df.get("IC", ic_df.get("pearson_ic"))
    avg_ic_val = float(avg_ic.mean()) if avg_ic is not None else float("nan")
    top_factor = (
        ic_df.iloc[0]["factor"]
        if "factor" in ic_df.columns and not ic_df.empty
        else "N/A"
    )

    if attribution_df is not None and not attribution_df.empty:
        best_year = attribution_df["sum"].idxmax()
        worst_year = attribution_df["sum"].idxmin()
        attr_line = f"Attribution strongest in {best_year} and weakest in {worst_year}."
    else:
        attr_line = "Attribution summary unavailable."

    return (
        f"IC average = {avg_ic_val:.3f}; leading factor = {top_factor}. "
        f"{attr_line}"
    )


__all__ = ["generate_llm_summary"]
