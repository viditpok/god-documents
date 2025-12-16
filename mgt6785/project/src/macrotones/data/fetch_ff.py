import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

CFG = yaml.safe_load(open("config/project.yaml"))
RAW = Path(CFG["data"]["out_raw"]).joinpath("ff")
RAW.mkdir(parents=True, exist_ok=True)

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _read_zip_csv(url: str) -> str:
    """Download a Ken French zip and return the CSV text inside."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # pick the first CSV-like file
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            return fh.read().decode("latin-1")  # encoding on site is latin-1


def _extract_monthly_block(csv_text: str) -> pd.DataFrame:
    """
    Extract the monthly data block robustly from Ken French CSV.

    Robust parsing rules:
    1. Locate first row that looks like YYYYMM (e.g., 192607)
    2. Use the immediately preceding non-empty line as headers if it contains non-numeric cells
    3. The monthly block ends at first blank line OR line starting with "Annual"
    4. Handle BOMs, whitespace, CRLF vs LF, multiple spaces/commas
    """
    # Clean up the text: strip BOM, normalize line endings, remove excessive whitespace
    csv_text = csv_text.strip()
    if csv_text.startswith("\ufeff"):  # Remove BOM
        csv_text = csv_text[1:]

    lines = [line.strip() for line in csv_text.splitlines()]

    # Find the first monthly data row (YYYYMM pattern)
    monthly_start = None
    header_row = None

    for i, line in enumerate(lines):
        if not line:  # Skip empty lines
            continue

        # Check if line starts with YYYYMM pattern
        if re.match(r"^\s*\d{6}\s*(,|;)", line):
            monthly_start = i
            # Look backwards for header row
            for j in range(i - 1, -1, -1):
                if lines[j] and not re.match(r"^\s*\d", lines[j]):  # Non-numeric line
                    header_row = j
                    break
            break

    if monthly_start is None:
        raise ValueError("Could not find monthly data block. No YYYYMM pattern found.")

    # Find the end of monthly data
    monthly_end = None
    for i in range(monthly_start + 1, len(lines)):
        line = lines[i]
        if not line:  # Blank line ends the block
            monthly_end = i
            break
        if re.match(r"^\s*Annual", line, re.IGNORECASE):  # "Annual" line ends the block
            monthly_end = i
            break

    if monthly_end is None:
        monthly_end = len(lines)

    # Extract the data block
    data_lines = lines[monthly_start:monthly_end]

    if len(data_lines) == 0:
        raise ValueError("No monthly data found in the extracted block.")

    # Prepare CSV content
    csv_content = []

    # Add header if we found one
    if header_row is not None:
        # Clean up header line - handle various separators and normalize
        header_line = lines[header_row]
        # Split by comma or semicolon, strip whitespace
        headers = re.split(r"[,;]", header_line)
        headers = [h.strip() for h in headers]
        csv_content.append(",".join(headers))

    # Add data lines
    for line in data_lines:
        # Clean up data line - normalize separators to commas
        cleaned_line = re.sub(r"[,;]\s*", ",", line.strip())
        csv_content.append(cleaned_line)

    # Parse with pandas
    csv_text_clean = "\n".join(csv_content)

    try:
        df = pd.read_csv(io.StringIO(csv_text_clean))
    except Exception as e:
        # Fallback: try to parse manually if pandas fails
        print(f"Pandas parsing failed, trying manual parsing: {e}")
        df = _manual_parse_monthly_data(
            data_lines, headers if header_row is not None else None
        )

    # Normalize column names (strip spaces and handle common variations)
    df.columns = [c.strip() for c in df.columns]

    # Find the date column (first column that looks like dates)
    date_col = None
    for col in df.columns:
        # Check if this column contains YYYYMM patterns
        if df[col].astype(str).str.match(r"^\d{6}$").any():
            date_col = col
            break

    if date_col is None:
        raise ValueError("Could not identify date column in monthly data.")

    # Clean and filter date column
    df[date_col] = df[date_col].astype(str).str.strip()
    df = df[df[date_col].str.match(r"^\d{6}$")].copy()

    if len(df) == 0:
        raise ValueError("No valid monthly data rows found after filtering.")

    if len(df) < 200:
        raise ValueError(
            f"Too few monthly data rows found ({len(df)}). Expected at least 200. "
            "This might indicate a parsing issue."
        )

    # Convert to proper datetime index
    dates = pd.to_datetime(df[date_col], format="%Y%m")
    # Convert to month-end timestamps manually
    month_end_dates = dates + pd.offsets.MonthEnd(0)
    df.index = month_end_dates
    df = df.drop(columns=[date_col])

    return df


def _manual_parse_monthly_data(data_lines: list, headers: list = None) -> pd.DataFrame:
    """Fallback manual parsing if pandas fails."""
    if headers is None:
        # Auto-generate headers based on number of columns in first data row
        first_line_parts = re.split(r"[,;]", data_lines[0].strip())
        headers = ["Date"] + [f"Col_{i}" for i in range(1, len(first_line_parts))]

    rows = []
    for line in data_lines:
        parts = re.split(r"[,;]", line.strip())
        if len(parts) >= len(headers):
            # Pad with empty strings if needed
            while len(parts) < len(headers):
                parts.append("")
            rows.append(parts[: len(headers)])

    return pd.DataFrame(rows, columns=headers)


def get_ff3_monthly() -> pd.DataFrame:
    """Get FF3 monthly factors (Mkt_RF, SMB, HML, RF)."""
    print("Fetching FF3 monthly factors...")
    txt = _read_zip_csv(FF3_URL)
    df = _extract_monthly_block(txt)

    # Standardize column names: Mkt-RF -> Mkt_RF
    ren = {c: c.replace("Mkt-RF", "Mkt_RF") for c in df.columns}
    df = df.rename(columns=ren)

    # Keep only the columns we need
    keep = [c for c in ["Mkt_RF", "SMB", "HML", "RF"] if c in df.columns]
    if not keep:
        raise ValueError(
            f"None of the expected FF3 columns found. Available columns: {list(df.columns)}"
        )

    df = df[keep]

    # Convert from percent to decimal
    df = df / 100.0

    print(f"FF3 data: {df.shape}, columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def get_mom_monthly() -> pd.DataFrame:
    """Get Momentum monthly factor (UMD)."""
    print("Fetching Momentum monthly factor...")
    txt = _read_zip_csv(MOM_URL)
    df = _extract_monthly_block(txt)

    # Find the momentum column (usually the first numeric column)
    mom_col = None
    for col in df.columns:
        if col.lower() in ["mom", "momentum", "umd"] or col.strip().lower() in [
            "mom",
            "momentum",
            "umd",
        ]:
            mom_col = col
            break

    if mom_col is None and len(df.columns) > 0:
        # Take the first column if we can't find a clear momentum column
        mom_col = df.columns[0]

    if mom_col is None:
        raise ValueError(
            f"No momentum column found. Available columns: {list(df.columns)}"
        )

    # Rename to UMD
    df = df.rename(columns={mom_col: "UMD"})
    df = df[["UMD"]] / 100.0  # Convert from percent to decimal

    print(f"Momentum data: {df.shape}, columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def main():
    """Main function to fetch and combine FF3 and Momentum data."""
    print("Starting Fama-French data fetch...")

    try:
        ff3 = get_ff3_monthly()
        mom = get_mom_monthly()

        # Join the dataframes
        out = ff3.join(mom, how="left").sort_index()

        # Handle UMD NaNs
        out["UMD"] = out["UMD"].astype(float)
        # Prefer forward fill then back fill; if still NaN, drop those months in downstream joins
        out["UMD"] = out["UMD"].ffill().bfill()

        # Save to parquet
        out_path = RAW / "ff_monthly.parquet"
        out.to_parquet(out_path)

        print(f"\n✅ Successfully saved FF data: {out.shape} -> {out_path}")
        print(f"Columns: {list(out.columns)}")
        print(f"Date range: {out.index[0]} to {out.index[-1]}")

        # Show first and last few dates
        print(f"\nFirst 5 dates: {out.head().index.strftime('%Y-%m-%d').tolist()}")
        print(f"Last 5 dates: {out.tail().index.strftime('%Y-%m-%d').tolist()}")

        # Basic validation
        if len(out) < 200:
            print(f"⚠️  Warning: Only {len(out)} rows found. Expected at least 200.")

        # Check for missing values
        missing_counts = out.isnull().sum()
        if missing_counts.sum() > 0:
            print(
                f"⚠️  Warning: Found missing values:\n{missing_counts[missing_counts > 0]}"
            )
        else:
            print("✅ No missing values found.")

    except Exception as e:
        print(f"❌ Error fetching Fama-French data: {e}")
        raise


if __name__ == "__main__":
    main()
