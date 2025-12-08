from __future__ import annotations

import math

import pandas as pd

from macrotones.data.macro_loader import get_macro_features


def test_macro_loader_returns_expected_keys() -> None:
    date = pd.Timestamp("2016-12-31")
    features = get_macro_features(date)
    expected = {"cpi_yoy", "gdp_real", "y10y_2y_spread", "unemp", "mri"}
    assert expected <= features.keys()
    for key in expected:
        assert isinstance(features[key], float)
        assert not math.isnan(features[key])
    assert abs(features["mri"]) < 20.0
