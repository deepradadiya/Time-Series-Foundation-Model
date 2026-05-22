# WARNING: This dataset must NEVER be used during pretraining. It is only for zero-shot evaluation.
"""Download script for the ETTh1 benchmark dataset.

This module downloads the ETTh1 (Electricity Transformer Temperature - Hourly)
dataset from the ETDataset GitHub repository. The dataset is used exclusively
for zero-shot evaluation of the pretrained model and must NOT be included in
pretraining data.

Source: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

Related modules:
    - config.py: Provides MAX_RETRIES and RETRY_BASE_DELAY settings
    - data/preprocess_pipeline.py: Consumes the downloaded CSV for preprocessing
"""

import os
import time
import urllib.request
import urllib.error
from io import StringIO

import pandas as pd

from config import Config


# Canonical download URL for ETTh1 dataset
ETTH1_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)

# Minimum expected rows in the ETTh1 dataset
MIN_ROWS: int = 17_000


def download_etth1(output_path: str = "data/raw/etth1.csv") -> str:
    """Download ETTh1 dataset from ETDataset GitHub repository.

    Downloads ETTh1.csv, extracts the 'OT' (oil temperature) column as a
    univariate series, and renames the 'date' column to 'timestamp'.
    Saves as CSV with 'timestamp' and 'value' columns.

    This dataset is reserved for zero-shot evaluation only and must never
    be used during pretraining.

    Args:
        output_path: Path where the processed CSV will be saved.
            Defaults to "data/raw/etth1.csv".

    Returns:
        Path to the saved CSV file.

    Raises:
        RuntimeError: If download fails after MAX_RETRIES attempts.
        ValueError: If 'OT' column is missing or file has < 17,000 rows.
    """
    # Skip download if file already exists (Requirement 4.6)
    if os.path.exists(output_path):
        print(f"[SKIP] ETTh1 dataset already exists at: {output_path}")
        return output_path

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download with retry and exponential backoff (Requirement 4.5)
    last_error: Exception | None = None
    raw_content: bytes | None = None

    for attempt in range(Config.MAX_RETRIES):
        try:
            if attempt > 0:
                delay: float = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] Attempt {attempt + 1}/{Config.MAX_RETRIES} "
                    f"for ETTh1 after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            else:
                print(f"[DOWNLOAD] Downloading ETTh1 dataset from: {ETTH1_URL}")

            response = urllib.request.urlopen(ETTH1_URL, timeout=30)
            raw_content = response.read()
            print(f"[SUCCESS] ETTh1 dataset downloaded successfully.")
            break

        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as e:
            last_error = e
            print(
                f"[ERROR] Download attempt {attempt + 1}/{Config.MAX_RETRIES} "
                f"failed for ETTh1: {e}"
            )

    if raw_content is None:
        raise RuntimeError(
            f"Failed to download ETTh1 dataset from {ETTH1_URL} after "
            f"{Config.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    # Parse the downloaded CSV content
    df: pd.DataFrame = pd.read_csv(StringIO(raw_content.decode("utf-8")))

    # Verify 'OT' column exists (Requirement 4.7)
    if "OT" not in df.columns:
        # Delete any partially saved file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError(
            f"ETTh1 dataset is missing the expected 'OT' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract 'OT' column and rename 'date' to 'timestamp' (Requirement 4.2)
    result = pd.DataFrame({
        "timestamp": df["date"],
        "value": df["OT"],
    })

    # Save as CSV (Requirement 4.3)
    result.to_csv(output_path, index=False)

    # Verify minimum row count (Requirement 4.8)
    row_count: int = len(result)
    if row_count < MIN_ROWS:
        os.remove(output_path)
        raise ValueError(
            f"ETTh1 dataset has {row_count} rows, but at least "
            f"{MIN_ROWS} rows are required."
        )

    # Print statistics
    print(f"\n{'='*60}")
    print(f"  ETTh1 Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Series length: {row_count:,}")
    print(f"  Min value:     {result['value'].min():.4f}")
    print(f"  Max value:     {result['value'].max():.4f}")
    print(f"  Mean value:    {result['value'].mean():.4f}")
    print(f"  NaN count:     {result['value'].isna().sum()}")
    print(f"  Saved to:      {output_path}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    download_etth1()
