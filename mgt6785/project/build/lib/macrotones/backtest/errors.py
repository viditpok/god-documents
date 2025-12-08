from __future__ import annotations


class BacktestError(RuntimeError):
    """Raised when the backtest encounters irrecoverable issues."""


__all__ = ["BacktestError"]
