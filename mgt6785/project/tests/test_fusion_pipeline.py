from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrotones.core.weights import blend_with_llm
from macrotones.reports.logs import append_policy_log


def test_blend_with_llm_and_logging(monkeypatch, tmp_path) -> None:
    date = pd.Timestamp("2020-06-30")
    w_ridge = pd.Series({"Mkt_RF": 0.4, "SMB": 0.3, "HML": 0.2, "UMD": 0.1})
    macro = {
        "cpi_yoy": 2.0,
        "gdp_real": 2.5,
        "y10y_2y_spread": 1.0,
        "unemp": 4.0,
        "mri": 0.8,
    }
    nlp = {
        "pos": 0.6,
        "neg": 0.2,
        "uncertainty": 0.1,
        "inflation_mention": 0.03,
        "growth_mention": 0.04,
        "nlp_regime": 0.35,
    }
    llm_weights = {"Mkt_RF": 0.5, "SMB": 0.2, "HML": 0.1, "UMD": 0.2}

    monkeypatch.setattr("macrotones.core.weights.get_macro_features", lambda d: macro)
    monkeypatch.setattr("macrotones.core.weights.get_nlp_regime", lambda d: nlp)
    monkeypatch.setattr(
        "macrotones.core.weights.generate_policy",
        lambda m, n: llm_weights,
    )
    monkeypatch.setattr(
        "macrotones.core.weights.get_last_rationale",
        lambda: "test rationale",
    )

    result = blend_with_llm(date=date, w_ridge=w_ridge, beta_neutral=False)
    expected = (1 - result.lambda_blend) * w_ridge + result.lambda_blend * pd.Series(
        llm_weights, index=w_ridge.index
    )
    assert np.allclose(result.w_final.values, expected.values)
    assert result.rationale == "test rationale"

    log_path = tmp_path / "llm_policy_log.csv"
    append_policy_log(result, log_path)
    df = pd.read_csv(log_path)
    assert df.shape[0] == 1
    assert "w_ridge_Mkt_RF" in df.columns
    assert df.loc[0, "lambda"] == pytest.approx(result.lambda_blend)
