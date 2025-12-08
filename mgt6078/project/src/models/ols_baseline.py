"""
ols baseline replicator for peer-implied fair value mapping to bartram & grinblatt (2018) section 3 methodology
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import PeerRegressor


class OLSBaseline(PeerRegressor):
    """
    ordinary least squares implementation
    """

    # guideline §5 empirical strategy l37-l52

    def __init__(self) -> None:
        super().__init__()
        self.coef_: Optional[np.ndarray] = None
        self.feature_names: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OLSBaseline":
        """
        estimate coefficients via normal equations
        """

        if X.empty:
            raise ValueError("features frame is empty")
        if len(X) != len(y):
            raise ValueError("feature and target lengths mismatch")
        self.feature_names = list(X.columns)
        design = np.column_stack([np.ones(len(X)), X.to_numpy()])
        target = y.to_numpy()
        gram = design.T @ design
        inv = np.linalg.pinv(gram)
        self.coef_ = inv @ design.T @ target
        self.logger.info("ols fit completed with %s features", len(self.feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        produce fair value predictions
        """

        if self.coef_ is None or self.feature_names is None:
            raise RuntimeError("model is not fit")
        X = X[self.feature_names]
        design = np.column_stack([np.ones(len(X)), X.to_numpy()])
        preds = design @ self.coef_
        self.logger.info("ols predict generated %s records", len(preds))
        # bartram & grinblatt eq (8): p = x (xᵀx)⁻¹ xᵀ v
        return pd.Series(preds, index=X.index)
