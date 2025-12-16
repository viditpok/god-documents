from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DOC_SCORES_PATH: Final[Path] = Path("data/interim/nlp_doc_scores.parquet")
REGIME_SCORES_PATH: Final[Path] = Path("data/interim/nlp_regime_scores.parquet")
WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z']+")
DATE_RE: Final[re.Pattern[str]] = re.compile(r"(19|20)\d{2}[-_/]\d{2}[-_/]\d{2}")
INFLATION_TERMS: Final[set[str]] = {
    "inflation",
    "price",
    "prices",
    "wage",
    "wages",
    "labor",
}
GROWTH_TERMS: Final[set[str]] = {
    "growth",
    "expansion",
    "improve",
    "improving",
    "improvement",
    "productivity",
}


def _month_end(date: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    return ts.to_period("M").to_timestamp("M")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _infer_doc_month(value: object, path_hint: str | None = None) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return _month_end(value)
    if isinstance(value, (int, float)) and not np.isnan(value):
        return _month_end(pd.Timestamp(value))
    if isinstance(value, str) and value:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.notna(ts):
            return _month_end(pd.Timestamp(ts))
    target = path_hint or ""
    if target:
        match = DATE_RE.search(target)
        if match:
            ts = pd.to_datetime(match.group(0), errors="coerce")
            if pd.notna(ts):
                return _month_end(pd.Timestamp(ts))
    return None


@lru_cache(maxsize=1)
def _load_doc_scores() -> pd.DataFrame:
    if not DOC_SCORES_PATH.exists():
        return pd.DataFrame(
            columns=["doc_month", "pos", "neg", "neu", "path", "n_chunks"]
        )
    df = pd.read_parquet(DOC_SCORES_PATH)
    paths = df.get("path")
    doc_month = df.get("doc_month")
    inferred = []
    for idx in range(len(df)):
        inferred.append(
            _infer_doc_month(
                doc_month.iloc[idx] if doc_month is not None else None,
                str(paths.iloc[idx]) if paths is not None else None,
            )
        )
    df = df.assign(doc_month=inferred)
    df = df.dropna(subset=["doc_month"])
    df["doc_month"] = df["doc_month"].apply(_month_end)
    df = df.sort_values("doc_month")
    return df


@lru_cache(maxsize=1)
def _load_regime_scores() -> pd.DataFrame:
    if not REGIME_SCORES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(REGIME_SCORES_PATH)
    if "doc_month" in df.columns:
        df.index = pd.to_datetime(df["doc_month"])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _sentiment_frame() -> pd.DataFrame:
    docs = _load_doc_scores()
    pieces: list[pd.DataFrame] = []
    if not docs.empty:
        agg = (
            docs.groupby("doc_month")[["pos", "neg", "neu"]]
            .mean()
            .rename(columns={"neu": "uncertainty"})
            .sort_index()
        )
        pieces.append(agg)
    regime = _load_regime_scores()
    if not regime.empty:
        reg_df = pd.DataFrame(index=regime.index)
        if "pos_mean_filled" in regime:
            reg_df["pos"] = regime["pos_mean_filled"]
        elif "pos_mean" in regime:
            reg_df["pos"] = regime["pos_mean"]
        if "neg_mean_filled" in regime:
            reg_df["neg"] = regime["neg_mean_filled"]
        elif "neg_mean" in regime:
            reg_df["neg"] = regime["neg_mean"]
        if "neu_mean_filled" in regime:
            reg_df["uncertainty"] = regime["neu_mean_filled"]
        elif "neu_mean" in regime:
            reg_df["uncertainty"] = regime["neu_mean"]
        pieces.append(reg_df)
    if not pieces:
        return pd.DataFrame(columns=["pos", "neg", "uncertainty"])
    merged = pd.concat(pieces, axis=1)
    merged = merged.sort_index()
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    merged["pos"] = merged["pos"].ffill()
    merged["neg"] = merged["neg"].ffill()
    merged["uncertainty"] = merged["uncertainty"].ffill()
    if "pos" not in merged or "neg" not in merged:
        raise ValueError("Sentiment sources missing both pos and neg columns.")
    merged["uncertainty"] = merged["uncertainty"].fillna(
        np.clip(1.0 - merged["pos"] - merged["neg"], 0.0, 1.0)
    )
    merged = merged.dropna(subset=["pos", "neg"], how="all")
    return merged


def _mentions_frame() -> pd.DataFrame:
    docs = _load_doc_scores()
    if docs.empty or "path" not in docs.columns:
        return pd.DataFrame(columns=["inflation_mention", "growth_mention"])
    records: list[dict[str, float]] = []
    for row in docs.itertuples():
        path = Path(row.path)
        text = _read_text(path)
        if not text.strip():
            continue
        tokens = [tok.lower() for tok in WORD_RE.findall(text)]
        if not tokens:
            continue
        total = max(1, len(tokens))
        inflation_hits = sum(tok in INFLATION_TERMS for tok in tokens)
        growth_hits = sum(tok in GROWTH_TERMS for tok in tokens)
        records.append(
            {
                "doc_month": row.doc_month,
                "inflation_mention": inflation_hits / total,
                "growth_mention": growth_hits / total,
            }
        )
    if not records:
        return pd.DataFrame(columns=["inflation_mention", "growth_mention"])
    df = pd.DataFrame(records)
    grouped = df.groupby("doc_month").mean().sort_index()
    return grouped


@lru_cache(maxsize=1)
def _nlp_table() -> pd.DataFrame:
    sent = _sentiment_frame()
    mentions = _mentions_frame()
    if sent.empty:
        return pd.DataFrame(
            columns=["pos", "neg", "uncertainty", "inflation_mention", "growth_mention"]
        )
    combined = sent.join(mentions, how="left")
    combined["inflation_mention"] = combined["inflation_mention"].fillna(0.0)
    combined["growth_mention"] = combined["growth_mention"].fillna(0.0)
    return combined


def get_nlp_regime(date: pd.Timestamp) -> dict[str, float]:
    """
    Return NLP sentiment/regime features for the requested month-end date.
    """
    table = _nlp_table()
    if table.empty:
        raise ValueError("NLP regime table is empty; run finbert_scoring first.")
    ts = _month_end(pd.Timestamp(date))
    if ts not in table.index:
        eligible = table.loc[:ts]
        if eligible.empty:
            raise ValueError(f"No NLP observations available on/before {ts.date()}.")
        row = eligible.iloc[-1]
    else:
        row = table.loc[ts]
    pos = float(row.get("pos", np.nan))
    neg = float(row.get("neg", np.nan))
    unc = float(row.get("uncertainty", np.nan))
    infl = float(row.get("inflation_mention", 0.0))
    growth = float(row.get("growth_mention", 0.0))
    if not np.isfinite(unc):
        unc = max(0.0, 1.0 - pos - neg)
    unc = max(0.0, min(1.0, unc))
    nlp_regime = pos - neg - 0.5 * unc
    out = {
        "pos": float(pos),
        "neg": float(neg),
        "uncertainty": float(unc),
        "inflation_mention": float(infl),
        "growth_mention": float(growth),
        "nlp_regime": float(nlp_regime),
    }
    for key, value in out.items():
        if not np.isfinite(value):
            out[key] = 0.0
    return out
