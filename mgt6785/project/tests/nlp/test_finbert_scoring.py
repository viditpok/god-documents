from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from macrotones.nlp import finbert_scoring


class DummyTokenizer:
    def __init__(self) -> None:
        self.decode_lengths: list[int] = []
        self.model_max_length = 512

    def __call__(
        self, text: str, *, return_tensors: str | None = None, **_: Any
    ) -> Any:
        input_ids = list(range(len(text)))
        if return_tensors == "pt":
            tensor = torch.tensor([input_ids[:512] or [0]], dtype=torch.long)
            return {"input_ids": tensor}
        return {"input_ids": input_ids}

    def decode(self, ids: list[int], **_: Any) -> str:
        self.decode_lengths.append(len(ids))
        return "x" * len(ids)


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_position_embeddings=512)
        self._device = "cpu"

    def eval(self) -> DummyModel:
        return self

    def to(self, device: Any) -> DummyModel:
        self._device = device
        return self

    def __call__(self, **_: Any) -> SimpleNamespace:
        logits = torch.tensor([[1.0, 0.5, -0.5]], dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def test_chunk_by_tokens_never_exceeds_limit() -> None:
    tokenizer = DummyTokenizer()
    text = "a" * 1500
    list(finbert_scoring.chunk_by_tokens(tokenizer, text, token_limit=512))
    assert tokenizer.decode_lengths
    assert all(length <= 512 for length in tokenizer.decode_lengths)


def test_main_uses_cache_to_skip_rescoring(monkeypatch, tmp_path) -> None:
    tokenizer = DummyTokenizer()
    model = DummyModel()

    monkeypatch.setattr(
        finbert_scoring.AutoTokenizer,
        "from_pretrained",
        lambda *_, **__: tokenizer,
    )
    monkeypatch.setattr(
        finbert_scoring.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_, **__: model,
    )

    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    for path in (raw_dir, interim_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(finbert_scoring, "RAW", raw_dir)
    monkeypatch.setattr(finbert_scoring, "INT", interim_dir)
    monkeypatch.setattr(finbert_scoring, "DOC_CACHE", interim_dir / "doc.parquet")
    monkeypatch.setattr(finbert_scoring, "OUT", processed_dir / "monthly.parquet")

    doc_path = raw_dir / "2017-06-14-minutes.txt"
    doc_path.write_text("Policy statement content.", encoding="utf-8")

    call_counter = {"count": 0}

    def fake_score_document(*args: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        call_counter["count"] += 1
        return np.array([0.6, 0.3, 0.1]), 1

    monkeypatch.setattr(finbert_scoring, "score_document", fake_score_document)

    finbert_scoring.main()
    assert call_counter["count"] == 1

    finbert_scoring.main()
    assert call_counter["count"] == 1, "score_document should not run when cache hit"


def test_aggregate_monthly_generates_filled_columns() -> None:
    doc_df = pd.DataFrame(
        [
            {
                "path": "a",
                "sha1": "sha1-a",
                "pos": 0.8,
                "neu": 0.15,
                "neg": 0.05,
                "n_chunks": 2,
                "doc_month": pd.Timestamp("2024-01-31"),
            },
            {
                "path": "b",
                "sha1": "sha1-b",
                "pos": 0.2,
                "neu": 0.7,
                "neg": 0.1,
                "n_chunks": 2,
                "doc_month": pd.Timestamp("2024-03-31"),
            },
        ]
    )

    monthly = finbert_scoring.aggregate_monthly(doc_df)
    filled_cols = [col for col in monthly.columns if col.endswith("_filled")]

    assert filled_cols, "Expected _filled smoothing columns to be created"
    assert "pos_minus_neg" in monthly.columns
    assert (monthly["pos_minus_neg_filled"] != monthly["pos_minus_neg"]).any()


def test_parse_doc_date_handles_prefixed_filename() -> None:
    ts = finbert_scoring.parse_doc_date(Path("fomc_2005-01-31_0.txt"))
    assert ts == pd.Timestamp("2005-01-31")


def test_parse_doc_date_handles_month_only_pattern() -> None:
    ts = finbert_scoring.parse_doc_date(Path("minutes_2008-05_summary"))
    assert ts == pd.Timestamp("2008-05-31")
