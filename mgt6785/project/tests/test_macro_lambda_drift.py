from __future__ import annotations

import numpy as np
import pandas as pd

from macrotones.utils import regime


def test_macro_lambda_drift_aligns_with_inverse_variance() -> None:
    idx = pd.date_range("2015-01-31", periods=24, freq="ME")
    lam = pd.Series(np.linspace(0.2, 0.8, len(idx)), index=idx)
    macro_signal = pd.Series(np.sin(np.linspace(0, 3, len(idx))) + 1.5, index=idx)
    drift = regime.macro_lambda_drift(lam, macro_signal, window=3)
    assert not drift.empty
    # When λ follows macro inverse variance exactly, drift should be near zero
    implied = regime.inverse_variance(macro_signal, window=3).reindex(idx)
    zero_drift = regime.macro_lambda_drift(implied, macro_signal, window=3)
    assert zero_drift.abs().max() < 1e-6
