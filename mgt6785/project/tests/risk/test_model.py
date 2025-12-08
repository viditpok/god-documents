from __future__ import annotations

import numpy as np
import pandas as pd

from macrotones.risk import model


def test_risk_model_returns_psd_matrix() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(size=(200, 3)),
        columns=["A", "B", "C"],
    )
    cov = model.ew_cov(data, halflife=6)
    shrunk = model.risk_model(data, halflife=6)
    assert cov is not None
    assert shrunk is not None
    assert shrunk.shape == (3, 3)
    eigvals = np.linalg.eigvalsh(shrunk.values)
    assert np.all(eigvals >= -1e-8)


def test_ledoit_wolf_shrink_reduces_condition_number() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(150, 4))
    df = pd.DataFrame(raw, columns=list("ABCD"))
    cov = model.ew_cov(df, halflife=3)
    assert cov is not None
    cond_sample = np.linalg.cond(cov.values)
    shrunk = model.risk_model(df, halflife=3)
    assert shrunk is not None
    cond_shrunk = np.linalg.cond(shrunk.values)
    assert cond_shrunk <= cond_sample + 1e-6
