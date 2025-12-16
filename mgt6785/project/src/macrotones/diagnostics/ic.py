from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _align_series(
    preds: pd.Series,
    realized: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    preds = preds.astype(float)
    realized = realized.reindex(preds.index).astype(float)
    mask = preds.notna() & realized.notna()
    return preds[mask], realized[mask]


def _t_stat_for_corr(r: float, n: int) -> float:
    if n < 3 or math.isclose(abs(r), 1.0):
        return float("nan")
    denom = max(1e-12, 1 - r**2)
    return r * math.sqrt((n - 2) / denom)


def ic_summary(
    preds: pd.DataFrame,
    realized: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for factor in factors:
        if factor not in preds.columns or factor not in realized.columns:
            continue
        x, y = _align_series(preds[factor], realized[factor])
        df_pair = pd.DataFrame({"pred": x, "realized": y})
        assert not df_pair.isna().any().any(), "Aligned IC inputs contain NaNs"
        n = len(x)
        if n < 3:
            rows.append(
                {
                    "factor": factor,
                    "pearson_ic": np.nan,
                    "pearson_pvalue": np.nan,
                    "spearman_ic": np.nan,
                    "spearman_pvalue": np.nan,
                    "hit_rate": np.nan,
                    "t_stat": np.nan,
                    "t_pvalue": np.nan,
                    "significant": False,
                }
            )
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        hit_rate = float((np.sign(x) == np.sign(y)).mean())
        t_stat = _t_stat_for_corr(pearson_r, n)
        if math.isnan(t_stat):
            t_pvalue = np.nan
        else:
            t_pvalue = 2 * stats.t.sf(abs(t_stat), df=n - 2)
        rows.append(
            {
                "factor": factor,
                "pearson_ic": pearson_r,
                "pearson_pvalue": pearson_p,
                "spearman_ic": spearman_r,
                "spearman_pvalue": spearman_p,
                "hit_rate": hit_rate,
                "t_stat": t_stat,
                "t_pvalue": t_pvalue,
                "significant": bool(pearson_p < 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


__all__ = ["ic_summary"]
