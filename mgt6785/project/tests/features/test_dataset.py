from __future__ import annotations

import pandas as pd

from macrotones.features import dataset


def test_dataset_panel_alignment_and_lags(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    ff_dir = raw_dir / "ff"
    for path in (raw_dir, interim_dir, processed_dir, ff_dir):
        path.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")

    fred = pd.DataFrame(
        {
            "CPI": range(100, 106),
            "SPECIAL": range(200, 206),
            "DGS10": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "DGS1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )
    fred.to_parquet(raw_dir / "fred_monthly.parquet")

    ff = pd.DataFrame(
        {
            "HML": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "SMB": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "UMD": [-0.01, 0.0, 0.01, 0.02, 0.03, 0.04],
            "Mkt_RF": [0.05, 0.04, 0.03, 0.02, 0.01, 0.0],
            "RF": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
        },
        index=dates,
    )
    ff.to_parquet(ff_dir / "ff_monthly.parquet")

    nlp = pd.DataFrame(
        {
            "pos_minus_neg_filled": [0.5, 0.6, 0.2, 0.3, 0.4, 0.5],
            "pos_minus_neg": [0.45, 0.55, 0.25, 0.35, 0.0, 0.1],
            "cov_docs": [1.0, 2.0, 0.0, 3.0, 0.0, 1.0],
        },
        index=dates,
    )
    nlp.to_parquet(interim_dir / "nlp_regime_scores.parquet")

    cfg = {
        "data": {
            "out_raw": str(raw_dir),
            "out_interim": str(interim_dir),
            "out_processed": str(processed_dir),
        },
        "project": {"start": "2020-01-31", "end": "2020-06-30"},
        "lags": {"fred_months": 1, "SPECIAL": 2, "nlp_months": 1},
    }

    monkeypatch.setattr(dataset, "CFG", cfg)
    monkeypatch.setattr(dataset, "RAW", raw_dir)
    monkeypatch.setattr(dataset, "INT", interim_dir)
    monkeypatch.setattr(dataset, "PRO", processed_dir)

    dataset.main()

    panel = pd.read_parquet(processed_dir / "panel.parquet")

    shifted_targets = ff[["HML", "SMB", "UMD"]].shift(-1).dropna()
    pd.testing.assert_series_equal(
        panel.loc[shifted_targets.index, "HML"],
        shifted_targets["HML"],
        check_names=False,
        check_freq=False,
    )

    assert "pos_minus_neg" in panel.columns
    assert "pos_minus_neg_filled" in panel.columns
    assert "nlp_cov_docs" in panel.columns

    fred_global_shift = fred.shift(1)
    cpi_index = fred_global_shift.dropna().index.intersection(panel.index)
    pd.testing.assert_series_equal(
        panel.loc[cpi_index, "CPI"],
        fred_global_shift.loc[cpi_index, "CPI"],
        check_names=False,
        check_freq=False,
    )

    special_expected = fred["SPECIAL"].shift(3).dropna()
    special_index = special_expected.index.intersection(panel.index)
    pd.testing.assert_series_equal(
        panel.loc[special_index, "SPECIAL"],
        special_expected.loc[special_index],
        check_names=False,
        check_freq=False,
    )
