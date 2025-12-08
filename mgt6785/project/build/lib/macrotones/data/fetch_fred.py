import os, yaml, pandas as pd
from fredapi import Fred
from pathlib import Path

CFG = yaml.safe_load(open("config/project.yaml"))
RAW = Path(CFG["data"]["out_raw"])
RAW.mkdir(parents=True, exist_ok=True)


def _fred():
    api_key = os.getenv("FRED_API_KEY")
    return Fred(api_key=api_key)


def main():
    fred = _fred()
    out = {}
    for s in CFG["data"]["fred_series"]:
        ser = fred.get_series(s)
        out[s] = ser
    df = pd.DataFrame(out)

    # normalize to month-end
    df = df.asfreq("D").ffill().resample("ME").last()

    # YoY transforms
    if "CPIAUCSL" in df.columns:
        df["CPI_YoY"] = df["CPIAUCSL"].pct_change(12) * 100
    if "INDPRO" in df.columns:
        df["INDPRO_YoY"] = df["INDPRO"].pct_change(12) * 100

    out_path = RAW / "fred_monthly.parquet"
    df.to_parquet(out_path)
    print(f"Saved {df.shape} -> {out_path}")


if __name__ == "__main__":
    main()
