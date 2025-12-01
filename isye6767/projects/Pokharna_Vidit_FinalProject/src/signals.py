"""Trading signal logic."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TradingThresholds


@dataclass
class SignalDecision:
    """Outcome of evaluating an s-score."""

    action: str
    new_position: int


class SignalGenerator:
    """Applies the four-threshold rule to produce trading actions."""

    def __init__(self, thresholds: TradingThresholds) -> None:
        self.thresholds = thresholds

    def evaluate(self, s_score: float | None, current_position: int) -> SignalDecision:
        """Return the appropriate action for the supplied s-score."""
        if s_score is None:
            return SignalDecision(action="NO_DATA", new_position=current_position)

        action = "HOLD"
        new_position = current_position

        # Close existing exposures first.
        if current_position > 0 and s_score >= -self.thresholds.ssc:
            action = "CLOSE_LONG"
            new_position = 0
        elif current_position < 0 and s_score <= self.thresholds.sbc:
            action = "CLOSE_SHORT"
            new_position = 0

        # Evaluate entry opportunities only if flat after the exit check.
        if new_position == 0:
            if s_score <= -self.thresholds.sbo:
                action = "BUY_OPEN"
                new_position = 1
            elif s_score >= self.thresholds.sso:
                action = "SELL_OPEN"
                new_position = -1

        return SignalDecision(action=action, new_position=new_position)

