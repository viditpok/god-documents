"""
Lasso / PLS / XGBoost peer fair-value regressors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV

from src.models.base import PeerRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


class LassoFairValue(PeerRegressor):
    def __init__(
        self,
        alphas: Optional[list[float]] = None,
        max_iter: int = 5000,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.model = LassoCV(
            alphas=alphas,
            cv=5,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=None,
        )
        self.feature_names: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoFairValue":
        if X.empty:
            raise ValueError("features frame is empty")
        if len(X) != len(y):
            raise ValueError("feature and target lengths mismatch")
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self.logger.info("lasso fit completed with %s features", len(self.feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.feature_names is None:
            raise RuntimeError("model is not fit")
        preds = self.model.predict(X[self.feature_names].values)
        return pd.Series(preds, index=X.index)


class PLSFairValue(PeerRegressor):
    def __init__(self, max_components: int = 10) -> None:
        super().__init__()
        self.max_components = max_components
        self.model: Optional[PLSRegression] = None
        self.feature_names: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PLSFairValue":
        if X.empty:
            raise ValueError("features frame is empty")
        if len(X) != len(y):
            raise ValueError("feature and target lengths mismatch")
        comps = min(self.max_components, X.shape[1], len(X) - 1)
        if comps < 1:
            raise ValueError("not enough samples for PLS components")
        self.feature_names = list(X.columns)
        self.model = PLSRegression(n_components=comps, scale=False)
        self.model.fit(X.values, y.values)
        self.logger.info("pls fit completed with %s comps", comps)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("model is not fit")
        preds = self.model.predict(X[self.feature_names].values).ravel()
        return pd.Series(preds, index=X.index)


class XGBFairValue(PeerRegressor):
    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        if XGBRegressor is None:
            raise ImportError("xgboost is required for XGBFairValue")
        self.feature_names: Optional[list[str]] = None
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBFairValue":
        if X.empty:
            raise ValueError("features frame is empty")
        if len(X) != len(y):
            raise ValueError("feature and target lengths mismatch")
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self.logger.info("xgb fit completed with %s records", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.feature_names is None:
            raise RuntimeError("model is not fit")
        preds = self.model.predict(X[self.feature_names].values)
        return pd.Series(preds, index=X.index)
