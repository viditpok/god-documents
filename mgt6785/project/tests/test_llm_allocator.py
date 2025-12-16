from __future__ import annotations

import math
from collections.abc import Mapping

from macrotones.fusion.llm_allocator import generate_policy


def _assert_weights(weights: Mapping[str, float]) -> None:
    assert set(weights) == {"Mkt_RF", "SMB", "HML", "UMD"}
    total = sum(weights.values())
    assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    for value in weights.values():
        assert -0.5 <= value <= 0.8


def test_llm_allocator_fallback_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    macro = {
        "mri": 0.3,
        "y10y_2y_spread": 1.0,
        "gdp_real": 2.0,
        "cpi_yoy": 2.2,
        "unemp": 3.8,
    }
    nlp = {
        "nlp_regime": 0.1,
        "inflation_mention": 0.05,
        "growth_mention": 0.02,
        "pos": 0.6,
        "neg": 0.2,
        "uncertainty": 0.2,
    }
    weights_first = generate_policy(macro, nlp)
    weights_second = generate_policy(macro, nlp)
    _assert_weights(weights_first)
    _assert_weights(weights_second)
    assert weights_first == weights_second
