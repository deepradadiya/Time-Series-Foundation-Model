"""Dataset download module for the Time Series Foundation Model.

This module handles downloading, verifying, and reporting statistics for the four
datasets used in the project: Energy (electricity consumption), Weather (observations),
Finance (stock/crypto OHLCV), and ETTh1 (Electricity Transformer Temperature benchmark).
It provides retry logic with exponential backoff for unreliable network conditions,
integrity verification to catch corrupt downloads, and detailed statistics printing.

Related modules:
    - config.py: Provides MAX_RETRIES and RETRY_BASE_DELAY settings
    - data/preprocess.py: Consumes the downloaded CSV files for normalization and splitting
    - data/dataset.py: Wraps preprocessed data into PyTorch Dataset objects
"""

import os
import time
import urllib.request
import urllib.error
from typing import Optional

import pandas as pd

# Import centralized configuration for retry settings
from config import Config


# ---------------------------------------------------------------------------
# Dataset URL Registry
# Each dataset is defined with a name, download URL, target save directory,
# and minimum expected row count for verification.
# ---------------------------------------------------------------------------

# Dataset definitions: (name, url, save_directory, min_rows)
# Energy: UCI Individual Household Electric Power Consumption (hourly resampled)
ENERGY_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
)

# Weather: Jena Climate dataset (10-minute intervals, multivariate)
WEATHER_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
)

# Finance: ETTh2 used as a proxy for financial-style multivariate data
FINANCE_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
)

# ETTh1: The primary benchmark dataset for zero-shot evaluation
ETTH1_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


def download_dataset(name: str, url: str, save_dir: str) -> str:
    """Download a single dataset file with exponential backoff retry logic.

    This function attempts to download a CSV file from the given URL. If the
    download fails due to a network error, it retries up to MAX_RETRIES times
    with exponential backoff (delay doubles each attempt, starting at
    RETRY_BASE_DELAY seconds). If the file already exists at the target path,
    the download is skipped entirely.

    Args:
        name: Human-readable name of the dataset (e.g., "Energy", "Weather").
        url: Full URL to download the CSV file from.
        save_dir: Local directory path where the file should be saved.

    Returns:
        The full file path where the dataset was saved (or already exists).

    Raises:
        RuntimeError: If all retry attempts fail, with details about the
            dataset name, URL, and the last error encountered.
    """
    # Construct the target file path from the dataset name
    filename: str = f"{name.lower()}.csv"
    filepath: str = os.path.join(save_dir, filename)

    # Skip download if the file already exists (Requirement 2.7)
    if os.path.exists(filepath):
        print(f"[SKIP] {name} dataset already exists at: {filepath}")
        return filepath

    # Ensure the save directory exists before attempting download
    os.makedirs(save_dir, exist_ok=True)

    # Retry loop with exponential backoff (Requirement 2.6)
    # Starts at RETRY_BASE_DELAY seconds, doubles each attempt
    last_error: Optional[Exception] = None

    for attempt in range(Config.MAX_RETRIES):
        try:
            # Calculate delay for this attempt (0 for first attempt)
            if attempt > 0:
                # Exponential backoff: base_delay * 2^(attempt-1)
                # Attempt 1: 2s, Attempt 2: 4s, Attempt 3: 8s
                delay: float = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] Attempt {attempt + 1}/{Config.MAX_RETRIES} "
                    f"for {name} after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            else:
                print(f"[DOWNLOAD] Downloading {name} dataset from: {url}")

            # Perform the actual HTTP download with a 30-second timeout
            # The timeout prevents hanging on unresponsive servers
            response = urllib.request.urlopen(url, timeout=30)

            # Write the response content to the target file
            with open(filepath, "wb") as out_file:
                out_file.write(response.read())

            # Download succeeded — report success and return the path
            print(f"[SUCCESS] {name} dataset saved to: {filepath}")
            return filepath

        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as e:
            # Network or filesystem error — record and potentially retry
            last_error = e
            print(
                f"[ERROR] Download attempt {attempt + 1}/{Config.MAX_RETRIES} "
                f"failed for {name}: {e}"
            )

    # All retries exhausted — raise with full context (Requirement 2.6)
    raise RuntimeError(
        f"Failed to download dataset '{name}' from {url} after "
        f"{Config.MAX_RETRIES} attempts. Last error: {last_error}"
    )


def verify_dataset(filepath: str, min_rows: int = 1000) -> dict:
    """Verify dataset integrity and compute statistics.

    Checks that the downloaded CSV file contains a parseable header row and
    at least `min_rows` data rows. If verification fails, the corrupt file
    is deleted and an error is raised. On success, returns a dictionary of
    dataset statistics including row count, column count, date range,
    missing value percentage, and file size.

    Args:
        filepath: Full path to the CSV file to verify.
        min_rows: Minimum number of data rows required (default: 1000).

    Returns:
        A dictionary containing dataset statistics:
            - "name": Dataset filename (without extension)
            - "rows": Number of data rows
            - "columns": Number of columns
            - "date_range": Dict with "start" and "end" date strings
            - "missing_pct": Percentage of missing values (0-100)
            - "file_size_bytes": File size in bytes

    Raises:
        ValueError: If the file has fewer than min_rows data rows or cannot
            be parsed as a valid CSV with a header. The corrupt file is
            deleted before raising.
    """
    # Get file size before attempting to parse
    file_size: int = os.path.getsize(filepath)

    # Extract dataset name from the filepath for reporting
    dataset_name: str = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Attempt to read the CSV file — this validates the header row
        df: pd.DataFrame = pd.read_csv(filepath)
    except Exception as e:
        # File is not a valid CSV — delete it and report failure (Requirement 2.8)
        os.remove(filepath)
        raise ValueError(
            f"Verification failed for '{dataset_name}': Cannot parse CSV. "
            f"Error: {e}. Corrupt file deleted."
        )

    # Check minimum row count (Requirement 2.8)
    row_count: int = len(df)
    if row_count < min_rows:
        # File has too few rows — delete and report (Requirement 2.8)
        os.remove(filepath)
        raise ValueError(
            f"Verification failed for '{dataset_name}': Found {row_count} rows, "
            f"minimum required is {min_rows}. Corrupt file deleted."
        )

    # Compute column count
    col_count: int = len(df.columns)

    # Attempt to detect and parse the date column for date range reporting
    # Look for common date column names
    date_range: dict = {"start": "N/A", "end": "N/A"}
    date_columns: list = [
        col for col in df.columns
        if col.lower() in ("date", "datetime", "timestamp", "time", "ds")
    ]

    if date_columns:
        # Parse the first matching date column
        date_col: str = date_columns[0]
        try:
            dates = pd.to_datetime(df[date_col])
            date_range = {
                "start": str(dates.min()),
                "end": str(dates.max()),
            }
        except (ValueError, TypeError):
            # Date parsing failed — leave as N/A
            pass

    # Calculate missing value percentage across all cells
    total_cells: int = row_count * col_count
    missing_cells: int = int(df.isna().sum().sum())
    missing_pct: float = (missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

    # Build the statistics dictionary (Requirement 2.5)
    stats: dict = {
        "name": dataset_name,
        "rows": row_count,
        "columns": col_count,
        "date_range": date_range,
        "missing_pct": round(missing_pct, 4),
        "file_size_bytes": file_size,
    }

    return stats


def print_dataset_stats(stats: dict) -> None:
    """Print formatted dataset statistics to standard output.

    Displays a human-readable summary of the dataset including row count,
    column count, date range, missing value percentage, and file size.

    Args:
        stats: Dictionary of statistics as returned by verify_dataset().
    """
    # Print a formatted block of statistics (Requirement 2.5)
    print(f"\n{'='*60}")
    print(f"  Dataset: {stats['name']}")
    print(f"{'='*60}")
    print(f"  Rows:          {stats['rows']:,}")
    print(f"  Columns:       {stats['columns']}")
    print(f"  Date Range:    {stats['date_range']['start']} → {stats['date_range']['end']}")
    print(f"  Missing (%):   {stats['missing_pct']:.4f}%")
    print(f"  File Size:     {stats['file_size_bytes']:,} bytes")
    print(f"{'='*60}\n")


def download_all() -> None:
    """Download and verify all four datasets used in the project.

    Downloads Energy, Weather, Finance, and ETTh1 datasets to their
    respective directories under `data/raw/`. Each download uses retry
    logic with exponential backoff. After downloading, each file is
    verified for integrity (parseable header + minimum row count).
    Statistics are printed for each successfully verified dataset.

    The four datasets serve different purposes:
        - Energy: Pretraining domain (electricity consumption patterns)
        - Weather: Pretraining domain (meteorological observations)
        - Finance: Pretraining domain (financial time series patterns)
        - ETTh1: Zero-shot evaluation benchmark (not used in pretraining)

    Raises:
        RuntimeError: If any dataset download fails after all retries.
        ValueError: If any dataset fails verification (corrupt/too few rows).
    """
    # Determine the project root directory (parent of the data/ package)
    # This allows the script to work regardless of the current working directory
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the base raw data directory
    raw_dir: str = os.path.join(project_root, "data", "raw")

    # Define the ETTh1 subdirectory (separate per Requirement 2.4)
    etth1_dir: str = os.path.join(raw_dir, "etth1")

    # Dataset registry: (name, url, save_directory, min_rows_for_verification)
    # Each entry specifies where to download from and where to save
    datasets: list = [
        ("energy", ENERGY_URL, raw_dir, 1000),
        ("weather", WEATHER_URL, raw_dir, 1000),
        ("finance", FINANCE_URL, raw_dir, 1000),
        ("etth1", ETTH1_URL, etth1_dir, 1000),
    ]

    print("\n" + "=" * 60)
    print("  TIME SERIES FOUNDATION MODEL — Dataset Download")
    print("=" * 60 + "\n")

    # Track successfully downloaded datasets for summary
    successful: list = []
    failed: list = []

    for name, url, save_dir, min_rows in datasets:
        try:
            # Step 1: Download the dataset (with retry logic)
            filepath: str = download_dataset(name, url, save_dir)

            # Step 2: Verify the downloaded file's integrity
            stats: dict = verify_dataset(filepath, min_rows=min_rows)

            # Step 3: Print statistics for the verified dataset
            print_dataset_stats(stats)

            # Record success
            successful.append(name)

        except (RuntimeError, ValueError) as e:
            # Download or verification failed — report and continue with others
            print(f"\n[FAILED] {name}: {e}\n")
            failed.append(name)

    # Print final summary
    print("\n" + "=" * 60)
    print(f"  Download Summary: {len(successful)} succeeded, {len(failed)} failed")
    if successful:
        print(f"  Successful: {', '.join(successful)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Script entry point — allows running directly: python -m data.download
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_all()
