from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from macrotones.backtest import diagnostics


def _write_config(base_dir: str, cfg: dict) -> None:
    cfg_path = Path(base_dir) / "project.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream)


def _prepare_env(tmp_path) -> dict[str, str]:
    config_dir = tmp_path / "config"
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    ff_dir = raw_dir / "ff"
    for path in (config_dir, processed_dir, raw_dir, ff_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "config": str(config_dir),
        "processed": str(processed_dir),
        "raw": str(raw_dir),
        "ff": str(ff_dir),
    }


def test_diagnostics_ic_alignment(monkeypatch, tmp_path) -> None:
    paths = _prepare_env(tmp_path)
    cfg = {
        "data": {
            "out_processed": paths["processed"],
            "out_raw": paths["raw"],
        }
    }
    _write_config(paths["config"], cfg)

    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])

    ff = pd.DataFrame(
        {
            "HML": [0.1, -0.2, 0.3, -0.1],
            "SMB": [0.05, 0.01, -0.02, 0.03],
            "UMD": [0.0, 0.02, -0.01, 0.01],
            "Mkt_RF": [0.02, 0.02, 0.01, 0.0],
            "RF": [0.001, 0.001, 0.001, 0.001],
        },
        index=idx,
    )
    ff.to_parquet(f"{paths['ff']}/ff_monthly.parquet")

    preds = pd.DataFrame(
        {
            "HML": ff["HML"].shift(-1),
            "SMB": ff["SMB"].shift(-1),
            "UMD": [np.nan, 0.01, -0.02, np.nan],
            "Mkt_RF": ff["Mkt_RF"],  # should be excluded later
        },
        index=idx,
    )
    preds.to_parquet(f"{paths['processed']}/preds.parquet")

    monkeypatch.chdir(tmp_path)
    diagnostics.main()

    summary = pd.read_csv(f"{paths['processed']}/ic_summary.csv", index_col="factor")

    assert "Mkt_RF" not in summary.index
    assert summary.loc["HML", "pearson_ic"] > 0.99
    assert summary.loc["SMB", "hit_rate"] > 0.75
