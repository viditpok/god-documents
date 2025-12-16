from __future__ import annotations

import pandas as pd


def regime_commentary(row: pd.Series | dict[str, float]) -> str:
    mri = float(row.get("mri", 0.0))
    nlp_regime = float(row.get("nlp_regime", 0.0))
    if mri > 1.0 and nlp_regime > 0.5:
        return "Expansionary tone: market optimism and positive macro momentum."
    if mri < -1.0 and nlp_regime < -0.2:
        return "Contractionary regime: defensive bias, low λ favors Ridge model."
    if mri < 0 and nlp_regime > 0:
        return "Policy divergence: positive tone amid weak fundamentals."
    return "Neutral macro-NLP alignment."
