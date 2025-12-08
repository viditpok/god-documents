import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    Args:
        series: Price series or series to compute RSI on.
        window: Lookback window.
    Returns:
        RSI series (0-100).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # Default to neutral if undefined


def compute_realized_vol(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute Realized Volatility (Standard Deviation).
    Args:
        series: Return series.
        window: Lookback window.
    Returns:
        Volatility series.
    """
    return series.rolling(window=window).std().fillna(0.0)
