"""
logging configuration ensuring scripts emit to logs/run.log
"""

from __future__ import annotations

import logging
from pathlib import Path


LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "run.log"


def get_logger(name: str) -> logging.Logger:
    """
    central logger factory writing to shared file
    """

    # guideline §5 empirical strategy l37-l52
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.FileHandler(LOG_PATH)
        stream = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler.setFormatter(fmt)
        stream.setFormatter(fmt)
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.addHandler(stream)
    return logging.getLogger(name)
