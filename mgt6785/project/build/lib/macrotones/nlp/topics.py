from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

INFLATION_KEYWORDS = {"inflation", "price", "prices", "wage", "wages", "labor"}
CREDIT_KEYWORDS = {"credit", "loan", "bank", "banks", "lending", "debt", "funding"}


def _read_text(row) -> str:
    text = getattr(row, "text", None)
    if text is not None:
        return text
    path = Path(row.path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _label_topics(words: np.ndarray, components: np.ndarray) -> dict[str, int]:
    topic_labels: dict[str, int] = {}
    top_indices = np.argsort(components, axis=1)[:, ::-1][:, :20]
    for idx, word_ids in enumerate(top_indices):
        vocab = set(words[word_ids])
        if "inflation" not in topic_labels and vocab & INFLATION_KEYWORDS:
            topic_labels["inflation"] = idx
        if "credit" not in topic_labels and vocab & CREDIT_KEYWORDS:
            topic_labels["credit"] = idx
    return topic_labels


def compute_topic_intensity(doc_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty:
        return pd.DataFrame(columns=["topic_inflation", "topic_credit"])

    corpus: list[str] = []
    months: list[pd.Timestamp] = []
    for row in doc_df.itertuples():
        text = _read_text(row)
        if not text.strip():
            continue
        corpus.append(text)
        months.append(pd.to_datetime(row.doc_month))

    if len(corpus) < 5:
        return pd.DataFrame(columns=["topic_inflation", "topic_credit"])

    vectorizer = CountVectorizer(
        max_features=2000,
        stop_words="english",
        lowercase=True,
    )
    dtm = vectorizer.fit_transform(corpus)
    if dtm.shape[0] == 0:
        return pd.DataFrame(columns=["topic_inflation", "topic_credit"])
    lda = LatentDirichletAllocation(
        n_components=10,
        random_state=42,
        learning_method="batch",
    )
    topic_weights = lda.fit_transform(dtm)
    vocab = vectorizer.get_feature_names_out()
    labels = _label_topics(vocab, lda.components_)
    if not labels:
        return pd.DataFrame(columns=["topic_inflation", "topic_credit"])

    topics_df = pd.DataFrame(
        topic_weights,
        index=months,
        columns=list(range(topic_weights.shape[1])),
    )

    data = {}
    if "inflation" in labels:
        data["topic_inflation"] = topics_df[labels["inflation"]]
    if "credit" in labels:
        data["topic_credit"] = topics_df[labels["credit"]]

    if not data:
        return pd.DataFrame(columns=["topic_inflation", "topic_credit"])

    monthly = pd.DataFrame(data).groupby(level=0).mean().sort_index()
    return monthly
