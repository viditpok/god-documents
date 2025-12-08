from __future__ import annotations

import math

import pandas as pd

from macrotones.data.nlp_sentiment import get_nlp_regime


def test_nlp_regime_outputs_probabilities() -> None:
    date = pd.Timestamp("2015-12-31")
    regime = get_nlp_regime(date)
    keys = [
        "pos",
        "neg",
        "uncertainty",
        "inflation_mention",
        "growth_mention",
        "nlp_regime",
    ]
    for key in keys:
        assert key in regime
        assert isinstance(regime[key], float)
    assert 0.0 <= regime["pos"] <= 1.0
    assert 0.0 <= regime["neg"] <= 1.0
    assert 0.0 <= regime["uncertainty"] <= 1.0
    assert 0.0 <= regime["inflation_mention"] <= 1.0
    assert 0.0 <= regime["growth_mention"] <= 1.0
    assert math.isfinite(regime["nlp_regime"])
