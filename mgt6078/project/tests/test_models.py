"""
tests for peer regressors
"""

import unittest

import pandas as pd

from src.models.ols_baseline import OLSBaseline
from src.models.theilsen_peer import TheilSenPeerRegressor


class ModelTests(unittest.TestCase):
    """
    confirm baseline regression behaves
    """

    # guideline §5 empirical strategy l37-l52

    def test_ols_fit_predict_shape(self) -> None:
        """
        predictions should align with training index
        """

        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.0, 1.0, 0.5]})
        y = pd.Series([1.5, 2.0, 2.5])
        model = OLSBaseline().fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertListEqual(list(preds.index), list(X.index))

    def test_ols_predict_requires_fit(self) -> None:
        """
        predicting before fitting raises runtime error
        """

        model = OLSBaseline()
        with self.assertRaises(RuntimeError):
            model.predict(pd.DataFrame({"x1": [1.0]}))

    def test_theilsen_fit_predict_shape(self) -> None:
        """
        theil-sen outputs align with training index
        """

        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.0, 1.0, 0.5]})
        y = pd.Series([1.5, 2.0, 2.5])
        model = TheilSenPeerRegressor().fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertListEqual(list(preds.index), list(X.index))

    def test_theilsen_predict_requires_fit(self) -> None:
        """
        predicting before fitting theil-sen raises runtime error
        """

        model = TheilSenPeerRegressor()
        with self.assertRaises(RuntimeError):
            model.predict(pd.DataFrame({"x1": [1.0]}))


if __name__ == "__main__":
    unittest.main()
