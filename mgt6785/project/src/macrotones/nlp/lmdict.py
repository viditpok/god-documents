from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

LM_DEFAULT = {
    "positive": {
        "growth",
        "increase",
        "improve",
        "strength",
        "opportunity",
        "positive",
        "optimistic",
    },
    "negative": {
        "decline",
        "loss",
        "weakness",
        "negative",
        "risk",
        "downturn",
        "crisis",
    },
    "uncertainty": {
        "uncertain",
        "volatility",
        "doubt",
        "unstable",
        "ambiguous",
        "question",
    },
}
WORD_RE = re.compile(r"[A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _score_tokens(tokens: Iterable[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {"lm_pos": 0.0, "lm_neg": 0.0, "lm_uncertainty": 0.0}

    def _count(words: set[str]) -> float:
        return sum(counts[word] for word in words) / total

    return {
        "lm_pos": _count(LM_DEFAULT["positive"]),
        "lm_neg": _count(LM_DEFAULT["negative"]),
        "lm_uncertainty": _count(LM_DEFAULT["uncertainty"]),
    }


def score_documents(doc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in doc_df.itertuples():
        text = getattr(row, "text", None)
        if text is None:
            text_path = Path(row.path)
            text = text_path.read_text(encoding="utf-8", errors="ignore")
        tokens = _tokenize(text)
        scores = _score_tokens(tokens)
        scores["doc_month"] = pd.to_datetime(row.doc_month)
        rows.append(scores)
    return pd.DataFrame(rows)


def aggregate_monthly(doc_scores: pd.DataFrame) -> pd.DataFrame:
    if doc_scores.empty:
        return pd.DataFrame(columns=["lm_pos_z", "lm_neg_z", "lm_uncertainty_z"])

    grouped = (
        doc_scores.groupby("doc_month")[["lm_pos", "lm_neg", "lm_uncertainty"]]
        .mean()
        .sort_index()
    )
    z_scores = grouped.apply(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )
    z_scores = z_scores.rename(
        columns={
            "lm_pos": "lm_pos_z",
            "lm_neg": "lm_neg_z",
            "lm_uncertainty": "lm_uncertainty_z",
        }
    )
    return z_scores
