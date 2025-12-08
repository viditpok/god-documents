from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping

import numpy as np
import requests

FACTOR_ORDER = ("Mkt_RF", "SMB", "HML", "UMD")
LOWER_BOUND = -0.5
UPPER_BOUND = 0.8
LAST_RATIONALE: str | None = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _project_weights(weights: Mapping[str, float]) -> dict[str, float]:
    keys = list(FACTOR_ORDER)
    values = np.array(
        [float(weights.get(key, 0.0)) for key in keys],
        dtype=float,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(values.sum())
    if abs(total) < 1e-9:
        values = np.full_like(values, 1.0 / len(values))
    else:
        values /= total
    for _ in range(100):
        over = values > UPPER_BOUND
        under = values < LOWER_BOUND
        if not over.any() and not under.any():
            break
        if over.any():
            excess = (values[over] - UPPER_BOUND).sum()
            values[over] = UPPER_BOUND
            avail = ~over
            if avail.any():
                values[avail] += excess / avail.sum()
        if under.any():
            deficit = (LOWER_BOUND - values[under]).sum()
            values[under] = LOWER_BOUND
            avail = ~under
            if avail.any():
                values[avail] -= deficit / avail.sum()
        total = float(values.sum())
        if abs(total) < 1e-9:
            values = np.full_like(values, 1.0 / len(values))
        else:
            values /= total
    total = float(values.sum())
    if abs(total) < 1e-9:
        values = np.full_like(values, 1.0 / len(values))
    else:
        values /= total
    values = np.clip(values, LOWER_BOUND, UPPER_BOUND)
    total = float(values.sum())
    if abs(total) < 1e-9:
        values = np.full_like(values, 1.0 / len(values))
    else:
        values /= total
    return {key: float(val) for key, val in zip(keys, values, strict=False)}


def _deterministic_policy(
    macro: Mapping[str, float],
    nlp: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    mri = float(macro.get("mri", 0.0))
    nlp_regime = float(nlp.get("nlp_regime", 0.0))
    spread = float(macro.get("y10y_2y_spread", 0.0))
    inflation = float(nlp.get("inflation_mention", 0.0))
    growth = float(nlp.get("growth_mention", 0.0))

    macro_sig = _sigmoid(mri)
    nlp_sig = _sigmoid(nlp_regime)
    risk_on = macro_sig + nlp_sig - 1.0
    spread_bias = math.tanh(spread / 3.0)
    inflation_bias = math.tanh(inflation * 6.0)
    growth_bias = math.tanh(growth * 6.0)

    raw = {
        "Mkt_RF": 0.35 + 0.25 * risk_on - 0.10 * inflation_bias,
        "SMB": 0.2 + 0.10 * growth_bias + 0.05 * spread_bias,
        "HML": 0.2 - 0.15 * risk_on + 0.15 * spread_bias + 0.05 * inflation_bias,
        "UMD": 0.25 + 0.20 * risk_on + 0.10 * growth_bias,
    }
    weights = _project_weights(raw)
    rationale = (
        "Deterministic fallback using macro MRI and NLP regime "
        "signals to tilt exposures."
    )
    return weights, rationale


def _call_openai(
    macro: Mapping[str, float],
    nlp: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a disciplined macro allocator. "
                    "Respond with JSON containing 'weights' and 'rationale'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given macro "
                    f"{dict(macro)} "
                    f"and NLP {dict(nlp)}, return target weights for "
                    "Mkt_RF, SMB, HML, UMD as normalized floats between -0.5 and 0.8."
                ),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response missing choices.")
    content = choices[0]["message"]["content"]
    parsed = json.loads(content)
    weights = parsed.get("weights", parsed)
    rationale = parsed.get("rationale") or ""
    projected = _project_weights(weights)
    return projected, rationale


def generate_policy(
    macro: Mapping[str, float],
    nlp: Mapping[str, float],
) -> dict[str, float]:
    """
    Generate factor weights using the OpenAI API if available, otherwise fallback
    to a deterministic sigmoid-based policy.
    """
    global LAST_RATIONALE
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            weights, rationale = _call_openai(macro, nlp)
        except Exception:
            weights, rationale = _deterministic_policy(macro, nlp)
    else:
        weights, rationale = _deterministic_policy(macro, nlp)
    LAST_RATIONALE = rationale
    return weights


def get_last_rationale() -> str | None:
    return LAST_RATIONALE
