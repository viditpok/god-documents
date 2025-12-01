"""Simple long/short portfolio book-keeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class PortfolioState:
    """Snapshot of the book at a timestamp."""

    timestamp: pd.Timestamp
    cash: float
    holdings_value: float
    equity: float
    hourly_return: float


@dataclass
class PortfolioSimulator:
    """Tracks cash and 1-share positions for each token."""

    initial_capital: float
    positions: Dict[str, int] = field(default_factory=dict)
    cash: float = field(init=False)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def _price(self, prices: pd.Series, token: str) -> float:
        price = float(prices.get(token, float("nan")))
        return price

    def apply_trade(self, token: str, action: str, price: float) -> None:
        """Execute a 1-share trade at the provided price."""
        current = self.positions.get(token, 0)
        if action == "BUY_OPEN":
            if current != 0:
                return
            self.cash -= price
            self.positions[token] = 1
        elif action == "SELL_OPEN":
            if current != 0:
                return
            self.cash += price
            self.positions[token] = -1
        elif action == "CLOSE_LONG":
            if current <= 0:
                return
            self.cash += price
            self.positions[token] = 0
        elif action == "CLOSE_SHORT":
            if current >= 0:
                return
            self.cash -= price
            self.positions[token] = 0

    def holdings_value(self, prices: pd.Series) -> float:
        """Mark the positions to market."""
        value = 0.0
        for token, qty in self.positions.items():
            if qty == 0:
                continue
            price = prices.get(token)
            if price is None or pd.isna(price):
                continue
            value += qty * float(price)
        return value

    def equity(self, prices: pd.Series) -> float:
        """Total portfolio value."""
        return self.cash + self.holdings_value(prices)

