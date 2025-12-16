from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macrotones.backtest import diagnostics_alpha


def _setup_env(tmp_path: Path) -> dict[str, Path]:
    cfg_dir = tmp_path / "config"
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    ff_dir = raw / "ff"
    for path in (cfg_dir, processed, ff_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"cfg": cfg_dir, "processed": processed, "raw": raw, "ff": ff_dir}


def test_diagnostics_alpha_outputs(monkeypatch, tmp_path) -> None:
    paths = _setup_env(tmp_path)
    config_path = paths["cfg"] / "project.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: Test",
                "  start: 2015-01-31",
                "  end: 2016-12-31",
                "  rebalance: M",
                "  rf_col: RF",
                "data:",
                f"  out_processed: {paths['processed']}",
                f"  out_raw: {paths['raw']}",
            ]
        ),
        encoding="utf-8",
    )

    dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    preds = pd.DataFrame(
        {
            "HML": np.linspace(0.01, 0.03, len(dates)),
            "SMB": np.linspace(-0.02, 0.01, len(dates)),
            "UMD": np.sin(np.arange(len(dates)) / 3) * 0.01,
            "Mkt_RF": np.random.default_rng(0).normal(0.01, 0.002, len(dates)),
            "RF": np.full(len(dates), 1.2),
        },
        index=dates,
    )
    preds.to_parquet(paths["processed"] / "preds.parquet")

    ff = pd.DataFrame(
        {
            "HML": np.random.default_rng(1).normal(0.01, 0.02, len(dates)),
            "SMB": np.random.default_rng(2).normal(0.0, 0.015, len(dates)),
            "UMD": np.random.default_rng(3).normal(0.005, 0.01, len(dates)),
            "Mkt_RF": np.random.default_rng(4).normal(0.01, 0.01, len(dates)),
            "RF": np.full(len(dates), 1.2),
        },
        index=dates,
    )
    ff.to_parquet(paths["ff"] / "ff_monthly.parquet")

    bt = pd.DataFrame(
        {
            "net_ret": np.random.default_rng(5).normal(0.01, 0.01, len(dates)),
            "rf": np.full(len(dates), 1.2 / 1200),
            "w_HML": np.linspace(0.2, 0.4, len(dates)),
            "w_SMB": np.linspace(0.3, 0.1, len(dates)),
            "w_UMD": np.linspace(0.5, 0.3, len(dates)),
            "ret_HML": ff["HML"],
            "ret_SMB": ff["SMB"],
            "ret_UMD": ff["UMD"],
        },
        index=dates,
    )
    bt.to_parquet(paths["processed"] / "backtest.parquet")

    monkeypatch.chdir(tmp_path)
    diagnostics_alpha.main()

    decay_path = paths["processed"] / "alpha_decay.csv"
    attrib_path = paths["processed"] / "attribution.csv"
    assert decay_path.exists()
    assert attrib_path.exists()

    decay = pd.read_csv(decay_path)
    assert set(decay["horizon"]).issubset({1, 2, 3, 4, 5, 6})
    assert {"factor", "horizon", "ic"} <= set(decay.columns)

    attrib = pd.read_csv(attrib_path)
    assert {"date", "factor", "pnl"} <= set(attrib.columns)
