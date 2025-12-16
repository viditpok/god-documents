from __future__ import annotations

import pandas as pd
from ui.tabs import simulator


def test_tornado_series_deterministic() -> None:
    macro = pd.Series({"CPI": 0.5, "UNRATE": -0.3, "T10Y2Y": 0.1})
    weights = pd.Series({"CPI": 0.2, "UNRATE": 0.5, "T10Y2Y": 0.3})
    first = simulator._tornado_series(0.4, macro, weights)
    second = simulator._tornado_series(0.4, macro, weights)
    pd.testing.assert_series_equal(first, second)
