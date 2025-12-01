"""Project configuration objects."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TradingThresholds:
    """Container for trading signal thresholds."""

    sbo: float = 1.25  # buy-to-open threshold (negative side)
    sso: float = 1.25  # sell-to-open threshold (positive side)
    sbc: float = 0.75  # close-short threshold
    ssc: float = 0.50  # close-long threshold


@dataclass
class ProjectConfig:
    """Holds file paths and model settings."""

    price_path: Path = Path("data/coin_all_prices_full.csv")
    universe_path: Path = Path("data/coin_universe_150K_40.csv")
    output_dir: Path = Path("outputs")
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    csv_dir: Path = field(default_factory=lambda: Path("outputs/csv"))
    window_hours: int = 240
    coverage_ratio: float = 0.80
    min_tokens: int = 10
    testing_start: datetime = datetime(2021, 9, 26, 0, 0, tzinfo=timezone.utc)
    testing_end: datetime = datetime(2022, 9, 25, 23, 0, tzinfo=timezone.utc)
    initial_capital: float = 100_000.0
    thresholds: TradingThresholds = field(default_factory=TradingThresholds)
    required_tokens: tuple[str, ...] = ("BTC", "ETH")

    def ensure_directories(self) -> None:
        """Create output folders if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
