"""Weather dataset download module for the Time Series Foundation Model.

This module downloads the Weather dataset (WTH.csv) from the PatchTST GitHub
repository, extracts the "OT" (oil temperature) column as a univariate series,
and saves it as a standardized CSV file for downstream preprocessing.

The WTH.csv file contains multivariate weather observations. This script
extracts only the "OT" column and maps the "date" column to "timestamp"
to produce the standard two-column format used by the preprocessing pipeline.

Related modules:
    - config.py: Provides MAX_RETRIES and RETRY_BASE_DELAY settings
    - data/preprocess_pipeline.py: Consumes the output CSV for normalization and splitting
"""

import io
import os
import time
import urllib.request
import urllib.error
from typing import Optional

import pandas as pd

from config import Config


# URL for the WTH.csv file in the PatchTST repository
WEATHER_URL: str = (
    "https://raw.githubusercontent.com/yuqinie98/PatchTST/"
    "main/PatchTST_supervised/dataset/WTH.csv"
)

# Connection timeout in seconds for HTTP requests
CONNECTION_TIMEOUT: int = 30

# Minimum number of rows required for a valid download
MIN_ROWS: int = 1000


def download_weather(output_path: str = "data/raw/weather.csv") -> str:
    """Download weather dataset from PatchTST GitHub repository.

    Downloads WTH.csv, extracts the 'OT' (oil temperature) column, and saves
    it as a two-column CSV with 'timestamp' and 'value' columns.

    Args:
        output_path: Path where the output CSV will be saved.
            Defaults to "data/raw/weather.csv".

    Returns:
        Path to the saved CSV file.

    Raises:
        RuntimeError: If download fails after MAX_RETRIES attempts.
        ValueError: If 'OT' column is missing or file has fewer than 1000 rows.
    """
    # Skip download if file already exists (Requirement 2.6)
    if os.path.exists(output_path):
        print(f"[SKIP] Weather dataset already exists at: {output_path}")
        return output_path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download with retry and exponential backoff (Requirement 2.5)
    raw_data: bytes = _download_with_retry(WEATHER_URL)

    # Parse the CSV content
    df = pd.read_csv(io.BytesIO(raw_data))

    # Validate that the "OT" column exists (Requirement 2.7)
    if "OT" not in df.columns:
        raise ValueError(
            f"Weather dataset does not contain expected 'OT' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract "OT" column and map "date" to "timestamp" (Requirement 2.2)
    result = pd.DataFrame()
    result["timestamp"] = df["date"]
    result["value"] = df["OT"]

    # Save the result (Requirement 2.3)
    result.to_csv(output_path, index=False)

    # Validate minimum row count (Requirement 2.8)
    row_count = len(result)
    if row_count < MIN_ROWS:
        # Delete the file and raise error for corrupt/incomplete download
        os.remove(output_path)
        raise ValueError(
            f"Weather dataset has only {row_count} rows, "
            f"minimum required is {MIN_ROWS}. File deleted."
        )

    # Print statistics (Requirement 2.4)
    _print_statistics(result)

    return output_path


def _download_with_retry(url: str) -> bytes:
    """Download data from URL with exponential backoff retry logic.

    Args:
        url: The URL to download from.

    Returns:
        Raw bytes of the downloaded content.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_error: Optional[Exception] = None

    for attempt in range(Config.MAX_RETRIES):
        try:
            if attempt > 0:
                # Exponential backoff: base_delay * 2^(attempt-1)
                delay: float = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] Attempt {attempt + 1}/{Config.MAX_RETRIES} "
                    f"for Weather after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            else:
                print(f"[DOWNLOAD] Downloading Weather dataset from: {url}")

            # HTTP GET with 30-second connection timeout (Requirement 2.1)
            response = urllib.request.urlopen(url, timeout=CONNECTION_TIMEOUT)
            data: bytes = response.read()

            print("[SUCCESS] Weather dataset downloaded successfully.")
            return data

        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as e:
            last_error = e
            print(
                f"[ERROR] Download attempt {attempt + 1}/{Config.MAX_RETRIES} "
                f"failed for Weather: {e}"
            )

    # All retries exhausted (Requirement 2.5)
    raise RuntimeError(
        f"Failed to download Weather dataset from {url} after "
        f"{Config.MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _print_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics to standard output.

    Prints series length, min, max, mean (4 decimal places), and NaN count.

    Args:
        df: DataFrame with 'value' column to compute statistics from.
    """
    values = df["value"]
    row_count = len(df)
    min_val = values.min()
    max_val = values.max()
    mean_val = values.mean()
    nan_count = int(values.isna().sum())

    print(f"\n{'='*60}")
    print(f"  Weather Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Series length: {row_count:,}")
    print(f"  Min value:     {min_val:.4f}")
    print(f"  Max value:     {max_val:.4f}")
    print(f"  Mean value:    {mean_val:.4f}")
    print(f"  NaN count:     {nan_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    download_weather()
