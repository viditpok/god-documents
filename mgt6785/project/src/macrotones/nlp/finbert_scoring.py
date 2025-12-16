from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from macrotones.nlp import lmdict, topics

RAW = Path("data/raw/fomc")
INT = Path("data/interim")
INT.mkdir(parents=True, exist_ok=True)
DOC_CACHE = INT / "nlp_doc_scores.parquet"
OUT = INT / "nlp_regime_scores.parquet"
MODEL = "yiyanghkust/finbert-tone"
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 200
FALLBACK_CHARS = 1_000
HALF_LIFE = 3  # months

try:
    CONFIG = yaml.safe_load(Path("config/project.yaml").read_text(encoding="utf-8"))
except FileNotFoundError:
    CONFIG = {}
NLP_BLEND = bool(CONFIG.get("nlp_blend", True))
DATE_PATTERN = re.compile(r"\d{4}[-_/]\d{2}[-_/]\d{2}")
MONTH_PATTERN = re.compile(r"\d{4}[-_/]\d{2}")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    end = ts.to_period("M").to_timestamp(how="end")
    return end.normalize()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_by_tokens(
    tokenizer: PreTrainedTokenizerBase, text: str, token_limit: int
) -> Iterator[str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if not input_ids:
        yield from chunk_by_chars(text)
        return

    chunk_len = min(CHUNK_TOKENS, token_limit)
    overlap = min(CHUNK_OVERLAP, max(chunk_len - 1, 1))
    stride = max(1, chunk_len - overlap)

    for start in range(0, len(input_ids), stride):
        segment = input_ids[start : start + chunk_len]
        if not segment:
            break
        decoded = tokenizer.decode(
            segment, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if decoded.strip():
            yield decoded


def chunk_by_chars(text: str) -> Iterator[str]:
    text = text.strip()
    if not text:
        return
    for start in range(0, len(text), FALLBACK_CHARS):
        chunk = text[start : start + FALLBACK_CHARS]
        if chunk.strip():
            yield chunk


def score_chunks(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    chunks: Iterable[str],
    device: torch.device,
    token_limit: int,
) -> np.ndarray:
    probs = []
    for chunk in chunks:
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=token_limit,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1)
        probs.append(prob.cpu().numpy()[0])
    return np.vstack(probs) if probs else np.empty((0, 3))


def load_doc_cache() -> pd.DataFrame:
    if DOC_CACHE.exists():
        return pd.read_parquet(DOC_CACHE)
    return pd.DataFrame(columns=["path", "sha1", "pos", "neu", "neg", "n_chunks"])


def parse_doc_date(path: Path) -> pd.Timestamp | None:
    stem = path.stem
    token: str | None = None
    match = DATE_PATTERN.search(stem)
    if match:
        token = match.group(0)
    else:
        month_match = MONTH_PATTERN.search(stem)
        if month_match:
            token = f"{month_match.group(0)}-01"
    if token is None:
        return None
    normalized = token.replace("_", "-").replace("/", "-")
    ts = pd.to_datetime(normalized, errors="coerce")
    if pd.isna(ts):
        return None
    return month_end(pd.Timestamp(ts))


def score_document(
    path: Path,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    token_limit: int,
) -> tuple[np.ndarray, int]:
    chunks = list(chunk_by_tokens(tokenizer, text, token_limit))
    if not chunks:
        chunks = list(chunk_by_chars(text))
    scores = score_chunks(tokenizer, model, chunks, device, token_limit)
    if scores.size == 0:
        raise ValueError(f"No usable chunks for {path}")
    return scores.mean(axis=0), scores.shape[0]


def aggregate_monthly(doc_df: pd.DataFrame) -> pd.DataFrame:
    doc_df = doc_df.copy()
    doc_df["doc_month"] = doc_df["doc_month"].astype("datetime64[ns]")
    agg = doc_df.groupby("doc_month").agg(
        pos_mean=("pos", "mean"),
        pos_median=("pos", "median"),
        neu_mean=("neu", "mean"),
        neu_median=("neu", "median"),
        neg_mean=("neg", "mean"),
        neg_median=("neg", "median"),
        n_docs=("path", "count"),
    )
    agg["cov_docs"] = agg["n_docs"]
    agg["cov_ok"] = agg["cov_docs"] >= 2
    agg["pos_minus_neg"] = agg["pos_mean"] - agg["neg_mean"]
    agg = agg.sort_index()
    agg.index = agg.index.normalize()
    agg["coverage_rolling_12m"] = agg["cov_docs"].rolling(12, min_periods=1).sum()

    alpha = 1 - 0.5 ** (1 / HALF_LIFE)
    fill_cols = [
        "pos_mean",
        "pos_median",
        "neu_mean",
        "neu_median",
        "neg_mean",
        "neg_median",
        "pos_minus_neg",
    ]
    for col in fill_cols:
        masked = agg[col].where(agg["cov_ok"])
        agg[f"{col}_filled"] = (
            masked.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().ffill()
        )
    return agg


def _add_blend_features(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    components = []
    if "pos_minus_neg" in monthly.columns:
        components.append(_zscore(monthly["pos_minus_neg"]).rename("finbert_z"))
    if {"lm_pos_z", "lm_neg_z"} <= set(monthly.columns):
        lm_component = monthly["lm_pos_z"] - monthly["lm_neg_z"]
        components.append(_zscore(lm_component).rename("lm_z"))
    if {"topic_inflation", "topic_credit"} <= set(monthly.columns):
        topic_component = monthly["topic_inflation"] - monthly["topic_credit"]
        components.append(_zscore(topic_component).rename("topic_z"))

    if not components:
        return monthly

    z_stack = pd.concat(components, axis=1)
    blend = z_stack.mean(axis=1)
    monthly["pos_minus_neg_blend"] = blend
    alpha = 1 - 0.5 ** (1 / HALF_LIFE)
    monthly["pos_minus_neg_blend_filled"] = (
        blend.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().ffill()
    )
    return monthly


def main() -> None:
    configure_logging()
    files = sorted(RAW.glob("*.txt"))
    if not files:
        logging.warning("No FOMC text files in %s. Run fetch_fomc first.", RAW)
        return

    cache_df = load_doc_cache()
    if os.environ.get("MACROTONES_SKIP_NLP") == "1":
        if cache_df.empty:
            logging.warning(
                "MACROTONES_SKIP_NLP=1 but NLP cache is empty; skipping aggregation."
            )
            return
        doc_df = cache_df.copy()
        if "doc_month" in doc_df.columns:
            doc_df["doc_month"] = pd.to_datetime(doc_df["doc_month"])
        else:
            doc_df["doc_month"] = [
                parse_doc_date(Path(p)) for p in doc_df["path"].astype(str)
            ]
        doc_df = doc_df.dropna(subset=["doc_month", "pos", "neu", "neg"])
        if doc_df.empty:
            logging.warning(
                "MACROTONES_SKIP_NLP=1 but cache lacks usable sentiment columns."
            )
            return
        monthly = aggregate_monthly(doc_df)
        lm_monthly = lmdict.aggregate_monthly(lmdict.score_documents(doc_df))
        topic_monthly = topics.compute_topic_intensity(doc_df)
        monthly = monthly.join(lm_monthly, how="left")
        monthly = monthly.join(topic_monthly, how="left")
        if NLP_BLEND:
            monthly = _add_blend_features(monthly)
        monthly.to_parquet(OUT)
        cache_df[["path", "sha1", "pos", "neu", "neg", "n_chunks"]].drop_duplicates(
            "sha1"
        ).to_parquet(DOC_CACHE, index=False)
        logging.info(
            "Skipping FinBERT scoring (MACROTONES_SKIP_NLP=1). Cached docs: %d.",
            doc_df.shape[0],
        )
        logging.info(
            "Saved NLP monthly tone: %s -> %s",
            monthly.shape,
            OUT,
        )
        logging.info(
            "Date span: %s → %s",
            monthly.index.min().date(),
            monthly.index.max().date(),
        )
        print(f"Saved NLP monthly tone: {monthly.shape} -> {OUT}")
        return

    cache_by_hash = {row.sha1: row for row in cache_df.itertuples()}

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(MODEL),  # type: ignore[no-untyped-call]
    )
    model = cast(
        PreTrainedModel,
        AutoModelForSequenceClassification.from_pretrained(MODEL),
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = cast(PreTrainedModel, cast(Any, model).to(device))
    else:
        model = cast(PreTrainedModel, cast(Any, model).to("cpu"))

    doc_rows = []
    token_limit = int(
        min(
            CHUNK_TOKENS,
            getattr(model.config, "max_position_embeddings", CHUNK_TOKENS),
            getattr(tokenizer, "model_max_length", CHUNK_TOKENS) or CHUNK_TOKENS,
        )
    )
    token_limit = max(16, token_limit)
    logging.info("FinBERT token limit set to %d tokens per chunk", token_limit)
    rescored = 0
    for path in files:
        text = read_text(path)
        digest = sha1_text(text)
        cached = cache_by_hash.get(digest)
        if cached is not None:
            pos, neu, neg, n_chunks = (
                cached.pos,
                cached.neu,
                cached.neg,
                int(cached.n_chunks),
            )
        else:
            probs, n_chunks = score_document(
                path, text, tokenizer, model, device, token_limit
            )
            pos, neu, neg = probs.tolist()
            rescored += 1
        doc_month = parse_doc_date(path)
        if doc_month is None:
            logging.warning("Skipping %s (unparsable date)", path)
            continue
        doc_rows.append(
            {
                "path": str(path),
                "sha1": digest,
                "pos": pos,
                "neu": neu,
                "neg": neg,
                "n_chunks": n_chunks,
                "doc_month": doc_month,
            }
        )

    if not doc_rows:
        logging.warning("No scored documents after filtering.")
        return

    doc_df = pd.DataFrame(doc_rows)
    doc_df = doc_df.dropna(subset=["pos", "neu", "neg"]).sort_values("doc_month")
    cache_out = doc_df[
        ["path", "sha1", "pos", "neu", "neg", "n_chunks"]
    ].drop_duplicates("sha1")
    cache_out.to_parquet(DOC_CACHE, index=False)

    monthly = aggregate_monthly(doc_df)
    lm_monthly = lmdict.aggregate_monthly(lmdict.score_documents(doc_df))
    topic_monthly = topics.compute_topic_intensity(doc_df)
    monthly = monthly.join(lm_monthly, how="left")
    monthly = monthly.join(topic_monthly, how="left")
    if NLP_BLEND:
        monthly = _add_blend_features(monthly)
    monthly.to_parquet(OUT)

    logging.info(
        "Rescored %d docs; cached %d docs total.", rescored, cache_out.shape[0]
    )
    logging.info(
        "Saved NLP monthly tone: %s -> %s",
        monthly.shape,
        OUT,
    )
    logging.info(
        "Date span: %s → %s", monthly.index.min().date(), monthly.index.max().date()
    )
    print(f"Saved NLP monthly tone: {monthly.shape} -> {OUT}")


if __name__ == "__main__":
    main()
