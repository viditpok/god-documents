import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CFG = yaml.safe_load(open("config/project.yaml"))
RAW = Path(CFG["data"]["out_raw"]).joinpath("etfs")
RAW.mkdir(parents=True, exist_ok=True)

# ETF Mapping based on "Smart Beta" research paper
ETF_MAP = {
    "VTV": "Value",
    "IJR": "Size",      # S&P SmallCap 600
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "USMV": "LowVol",
    "VUG": "Growth",
    "SPY": "Market"
}

def main():
    logging.info("Starting ETF data fetch...")
    
    tickers = list(ETF_MAP.keys())
    # Fetch monthly data (start from 2000 to be safe, or earlier)
    # auto_adjust=True accounts for splits/dividends
    data = yf.download(tickers, start="2000-01-01", interval="1mo", auto_adjust=True, progress=False)
    
    # 'data' is normally MultiIndex (Price, Ticker). We want 'Close' or 'Adj Close' (auto_adjust gives Close)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
        
    # Calculate monthly returns
    returns = closes.pct_change().dropna()
    
    # Rename columns to friendly names
    returns = returns.rename(columns=ETF_MAP)
    
    # Ensure Month-End Index (yfinance often gives start/mid dates)
    returns.index = pd.to_datetime(returns.index) + pd.offsets.MonthEnd(0)
    
    # Save
    out_path = RAW / "etfs_monthly.parquet"
    returns.to_parquet(out_path)
    
    logging.info(f"✅ Successfully saved ETF monthly returns: {returns.shape} -> {out_path}")
    logging.info(f"Columns: {list(returns.columns)}")
    logging.info(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")

if __name__ == "__main__":
    main()
