"""Finance dataset download module for the Time Series Foundation Model.

Downloads Bitcoin hourly close price data using the yfinance library.
Produces a standardized CSV with 'timestamp' and 'value' columns for
the preprocessing pipeline.

Related modules:
    - config.py: Provides MAX_RETRIES and RETRY_BASE_DELAY settings
    - data/preprocess_pipeline.py: Consumes the output CSV for normalization and splitting
"""

import os
import time

import pandas as pd
import yfinance as yf

from config import Config


def download_finance(output_path: str = "data/raw/finance.csv") -> str:
    """Download Bitcoin hourly close prices via yfinance.

    Uses ticker 'BTC-USD', interval '1h', period '2y'.
    Extracts 'Close' column, forward-fills NaN values before saving.
    Saves as CSV with 'timestamp' (ISO 8601) and 'value' columns.

    Args:
        output_path: Path where the CSV file will be saved.
            Defaults to 'data/raw/finance.csv'.

    Returns:
        Path to the saved CSV file.

    Raises:
        RuntimeError: If download fails after MAX_RETRIES attempts.
        ValueError: If fewer than 1000 rows returned.
    """
    # Skip download if file already exists (Requirement 3.6)
    if os.path.exists(output_path):
        print(f"[SKIP] Finance dataset already exists at: {output_path}")
        return output_path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Retry loop with exponential backoff (Requirement 3.5)
    last_error = None

    for attempt in range(Config.MAX_RETRIES):
        try:
            if attempt > 0:
                delay = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] Attempt {attempt + 1}/{Config.MAX_RETRIES} "
                    f"for Finance after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            else:
                print("[DOWNLOAD] Downloading Finance (BTC-USD) dataset via yfinance...")

            # Download Bitcoin hourly data (Requirement 3.1)
            df = yf.download(
                tickers="BTC-USD",
                interval="1h",
                period="2y",
                progress=False,
            )

            # Check if download returned valid data
            if df is None or df.empty:
                raise RuntimeError("yfinance returned empty DataFrame")

            # Handle MultiIndex columns (newer yfinance versions return
            # MultiIndex with (Price, Ticker) levels for single tickers)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)

            # Validate minimum row count (Requirement 3.7)
            if len(df) < 1000:
                raise ValueError(
                    f"Insufficient data: got {len(df)} rows, "
                    f"minimum required is 1000."
                )

            # Extract Close column (Requirement 3.2)
            close_series = df["Close"].copy()

            # Count original NaN values before forward-fill (Requirement 3.8)
            nan_count = int(close_series.isna().sum())

            # Forward-fill NaN values (Requirement 3.8)
            close_series = close_series.ffill()

            # Build output DataFrame with standardized columns (Requirement 3.3)
            output_df = pd.DataFrame({
                "timestamp": close_series.index.strftime("%Y-%m-%d %H:%M:%S"),
                "value": close_series.values,
            })

            # Save to CSV
            output_df.to_csv(output_path, index=False)

            # Print statistics (Requirement 3.4)
            print(f"[SUCCESS] Finance dataset saved to: {output_path}")
            print(f"  Series length: {len(output_df)}")
            print(f"  Min value: {output_df['value'].min():.2f}")
            print(f"  Max value: {output_df['value'].max():.2f}")
            print(f"  Mean value: {output_df['value'].mean():.2f}")
            print(f"  NaN values (original): {nan_count}")

            return output_path

        except (RuntimeError, OSError, KeyError, ConnectionError) as e:
            last_error = e
            print(
                f"[ERROR] Download attempt {attempt + 1}/{Config.MAX_RETRIES} "
                f"failed for Finance: {e}"
            )
        except ValueError:
            # ValueError for insufficient rows should not be retried
            raise

    # All retries exhausted (Requirement 3.5)
    raise RuntimeError(
        f"Failed to download Finance dataset after "
        f"{Config.MAX_RETRIES} attempts. Last error: {last_error}"
    )


if __name__ == "__main__":
    download_finance()
