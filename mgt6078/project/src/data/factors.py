from __future__ import annotations

from pathlib import Path

from src.data.loader import DataRequest, load_ff_factors
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data_processed"


def prepare_ff_factors(start: str = "1987-03-31", end: str = "2012-12-31") -> Path:
    """
    Prepare Ken French factor data for the sample window:

    - load ff.csv(.zip) via loader
    - parse dates to month-end
    - convert percent units to decimals
    - write to data_processed/factors/ff_factors.csv
    """

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    req = DataRequest(start=start, end=end)
    ff = load_ff_factors(req)

    out_dir = DATA_PROCESSED / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ff_factors.csv"

    ff.to_csv(out_path, index=False)
    logger.info("saved FF factors to %s", out_path)
    return out_path
