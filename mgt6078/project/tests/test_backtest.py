"""
tests for quintile backtest helper
"""

import unittest

import pandas as pd

from src.pipeline.backtest import compute_quintile_spreads


class BacktestTests(unittest.TestCase):
    """
    ensure quintile stats run for basic frame
    """

    # guideline §6 discussion l53-l74

    def test_quintile_spread_column(self) -> None:
        """
        q5 minus q1 spread should exist
        """

        frame = pd.DataFrame(
            {
                "date": ["2010-01-31"] * 5,
                "signal": [0.1, 0.2, 0.3, 0.4, 0.5],
                "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )
        result = compute_quintile_spreads(frame, "signal", "ret", "date")
        self.assertIn("q5_q1", result.columns)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
