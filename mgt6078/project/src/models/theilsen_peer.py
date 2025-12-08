"""
Theil–Sen peer regression implementation for robust fair-value estimation
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.linear_model import TheilSenRegressor

from src.models.base import PeerRegressor


class TheilSenPeerRegressor(PeerRegressor):
    """
    Robust peer regression via Theil–Sen median-of-slopes estimator.
    """

    # guideline §5 empirical strategy l37-l52

    def __init__(
        self,
        max_subpopulation: float = 1e4,
        n_subsamples: int | None = None,
        n_jobs: int | None = None,
        random_state: int | None = 0,
    ) -> None:
        super().__init__()
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: Optional[TheilSenRegressor] = None
        self.feature_names: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TheilSenPeerRegressor":
        """
        Fit the Theil–Sen regressor on the given design matrix.
        """

        if X.empty:
            raise ValueError("features frame is empty")
        if len(X) != len(y):
            raise ValueError("feature and target lengths mismatch")

        self.feature_names = list(X.columns)
        model = TheilSenRegressor(
            fit_intercept=True,
            max_subpopulation=self.max_subpopulation,
            n_subsamples=self.n_subsamples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        model.fit(X, y)
        self._model = model
        self.logger.info(
            "theil-sen fit completed (n=%d, features=%d)",
            len(X),
            len(self.feature_names),
        )
        return self

    @property
    def coef_(self) -> pd.Series:
        if self._model is None or self.feature_names is None:
            raise RuntimeError("model is not fit")
        return pd.Series(self._model.coef_, index=self.feature_names)

    @property
    def intercept_(self) -> float:
        if self._model is None:
            raise RuntimeError("model is not fit")
        return float(self._model.intercept_)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict peer-implied market capitalization.
        """

        if self._model is None or self.feature_names is None:
            raise RuntimeError("model is not fit")

        X = X[self.feature_names]
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index)
