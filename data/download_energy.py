"""Download the Energy (electricity hourly) dataset from HuggingFace.

This script downloads the UCI Electricity Load dataset from the HuggingFace
datasets hub ('monash_tsf/electricity_hourly'), extracts the first household
(index 0) with 100,000 time steps, and saves it as a standardized CSV file
for downstream preprocessing.

Related modules:
    - config.py: Provides MAX_RETRIES and RETRY_BASE_DELAY settings
    - data/preprocess_pipeline.py: Consumes the output CSV for normalization and splitting
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# Number of time steps to extract from the first household
REQUIRED_LENGTH: int = 100_000


def download_energy(output_path: str = "data/raw/energy.csv") -> str:
    """Download energy dataset from HuggingFace datasets.

    Uses 'monash_tsf/electricity_hourly' as primary source.
    Extracts first household (index 0), first 100,000 time steps.
    Saves as CSV with 'timestamp' and 'value' columns.

    Args:
        output_path: Path where the CSV file will be saved.
            Defaults to 'data/raw/energy.csv'.

    Returns:
        Path to saved CSV file.

    Raises:
        RuntimeError: If download fails after MAX_RETRIES attempts.
        ValueError: If series has fewer than 100,000 time steps.
    """
    # Resolve path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, output_path)

    # Skip download if file already exists (Requirement 1.6)
    if os.path.exists(filepath):
        print(f"[SKIP] Energy dataset already exists at: {filepath}")
        return filepath

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Retry with exponential backoff (Requirement 1.5)
    last_error = None
    for attempt in range(Config.MAX_RETRIES):
        try:
            if attempt > 0:
                delay = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] Attempt {attempt + 1}/{Config.MAX_RETRIES} "
                    f"for Energy after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            else:
                print("[DOWNLOAD] Downloading Energy dataset from HuggingFace...")

            # Import datasets here to isolate import errors from retry logic
            from datasets import load_dataset

            # Load the electricity hourly dataset from HuggingFace
            dataset = load_dataset(
                "monash_tsf",
                "electricity_hourly",
                split="train",
                trust_remote_code=True,
            )

            # Extract first household (index 0)
            first_household = dataset[0]

            # Get the time series values
            values = np.array(first_household["target"], dtype=np.float64)

            # Get the start timestamp
            start_timestamp = first_household["start"]

            # Validate length (Requirement 1.7)
            if len(values) < REQUIRED_LENGTH:
                raise ValueError(
                    f"Energy dataset has {len(values)} time steps, "
                    f"but {REQUIRED_LENGTH} are required."
                )

            # Take first 100,000 time steps (Requirement 1.2)
            values = values[:REQUIRED_LENGTH]

            # Generate hourly timestamps starting from the start date
            timestamps = pd.date_range(
                start=start_timestamp,
                periods=REQUIRED_LENGTH,
                freq="h",
            )

            # Create DataFrame with standardized columns (Requirement 1.3)
            df = pd.DataFrame({
                "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S"),
                "value": values,
            })

            # Save to CSV
            df.to_csv(filepath, index=False)

            print(f"[SUCCESS] Energy dataset saved to: {filepath}")

            # Print statistics (Requirement 1.4)
            _print_statistics(df)

            return filepath

        except (OSError, ConnectionError, TimeoutError, Exception) as e:
            # Catch ValueError separately — don't retry on data validation errors
            if isinstance(e, ValueError):
                raise
            last_error = e
            print(
                f"[ERROR] Download attempt {attempt + 1}/{Config.MAX_RETRIES} "
                f"failed for Energy: {e}"
            )

    # All retries exhausted (Requirement 1.5)
    raise RuntimeError(
        f"Failed to download Energy dataset after "
        f"{Config.MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _print_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics to standard output.

    Displays series length, min, max, mean (to 4 decimal places),
    and NaN count.

    Args:
        df: DataFrame with 'value' column.
    """
    values = df["value"]
    length = len(values)
    min_val = values.min()
    max_val = values.max()
    mean_val = values.mean()
    nan_count = int(values.isna().sum())

    print(f"\n{'='*60}")
    print(f"  Energy Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Length:    {length}")
    print(f"  Min:       {min_val:.4f}")
    print(f"  Max:       {max_val:.4f}")
    print(f"  Mean:      {mean_val:.4f}")
    print(f"  NaN count: {nan_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    download_energy()
