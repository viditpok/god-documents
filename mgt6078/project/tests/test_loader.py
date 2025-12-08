"""
tests for data loader scaffolding
"""

import unittest

from src.data.loader import COMPUSTAT_FEATURES, DataRequest, load_synthetic_compustat, load_synthetic_crsp


class LoaderTests(unittest.TestCase):
    """
    verify synthetic loaders preserve schema
    """

    # guideline §4 data sources l18-l36

    def test_compustat_synthetic_columns(self) -> None:
        """
        expect all 28 variables present
        """

        request = DataRequest(start="2010-01", end="2010-02")
        frame = load_synthetic_compustat(request, seed=1)
        self.assertFalse(frame.empty)
        for col in COMPUSTAT_FEATURES:
            self.assertIn(col, frame.columns)

    def test_crsp_synthetic_basic_shape(self) -> None:
        """
        expect core price columns available
        """

        request = DataRequest(start="2010-01", end="2010-02")
        frame = load_synthetic_crsp(request, seed=2)
        self.assertFalse(frame.empty)
        self.assertIn("ret", frame.columns)
        self.assertIn("permno", frame.columns)


if __name__ == "__main__":
    unittest.main()
