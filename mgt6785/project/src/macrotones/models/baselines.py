from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_model(alpha: float = 3.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=42)),
        ]
    )


def fit_ridge(Xtr: pd.DataFrame, ytr: pd.Series, alpha: float = 3.0) -> Pipeline:
    model = ridge_model(alpha=alpha)
    model.fit(Xtr, ytr)
    return model


def predict_ridge(model: Pipeline, Xte: pd.DataFrame) -> float:
    return float(model.predict(Xte)[0])


def fit_predict_ridge(
    Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame, alpha: float = 3.0
) -> float:
    model = fit_ridge(Xtr, ytr, alpha=alpha)
    return predict_ridge(model, Xte)
