from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from macrotones.core.weights import HybridWeightResult

DEFAULT_LOG_PATH = Path("data/processed/llm_policy_log.csv")


def _factor_columns(prefix: str, factors: Iterable[str]) -> dict[str, float]:
    return {f"{prefix}_{factor}": float(value) for factor, value in factors}


def append_policy_log(
    result: HybridWeightResult,
    path: Path | None = None,
) -> Path:
    """
    Append a single row with ridge/LLM/final weights and rationale.
    """
    log_path = path or DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        "date": result.date,
        "lambda": float(result.lambda_blend),
        "mri": float(result.macro.get("mri", 0.0)),
        "nlp_regime": float(result.nlp.get("nlp_regime", 0.0)),
        "rationale": result.rationale or "",
    }
    row.update(_factor_columns("w_ridge", result.w_ridge.items()))
    row.update(_factor_columns("w_llm", result.w_llm.items()))
    row.update(_factor_columns("w_final", result.w_final.items()))
    df = pd.DataFrame([row])
    df["date"] = pd.to_datetime(df["date"])
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)
    return log_path
