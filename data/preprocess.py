"""Data preprocessing module for the Time Series Foundation Model.

This file handles normalization (z-score), chronological train/val/test splitting,
and filtering of time series data. It computes statistics only from the training
split to prevent data leakage, and saves those statistics as JSON for later use
during inverse normalization at evaluation time.

Related modules:
    - config.py provides split ratios (TRAIN_RATIO, VAL_RATIO, TEST_RATIO) and
      CONTEXT_LENGTH (512) used as the minimum series length threshold.
    - data/patching.py consumes the normalized, split arrays produced here.
    - forecasting/inference.py uses inverse_normalize to return predictions to
      the original data scale.
    - evaluation/ uses the saved JSON stats for metric computation in original scale.
"""

import json
import logging
import os
import warnings
from typing import Any

import numpy as np

from config import Config

# Set up module-level logger for warnings about discarded series and zero-std channels
logger = logging.getLogger(__name__)


def compute_normalization_stats(train_data: np.ndarray) -> dict[str, list[float]]:
    """Compute per-channel mean and standard deviation from the training split.

    This function calculates statistics used for z-score normalization. Only the
    training data is used to avoid data leakage into validation/test sets.

    If a channel has zero standard deviation (constant values), the std is set
    to 1.0 to avoid division by zero, and a warning is logged.

    Parameters:
        train_data: A 2D numpy array of shape (time_steps, num_channels) containing
                    the training portion of the time series data.

    Returns:
        A dictionary with keys "mean" and "std", each mapping to a list of
        per-channel float values. For example:
        {"mean": [3.45, 7.89], "std": [1.23, 2.01]}
    """
    # Ensure input is 2D — if 1D, treat as single channel
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)

    # Compute mean along the time axis (axis=0) for each channel
    channel_means = np.mean(train_data, axis=0).tolist()

    # Compute standard deviation along the time axis for each channel
    channel_stds = np.std(train_data, axis=0).tolist()

    # Handle zero-std channels: set std to 1.0 to prevent division by zero
    for i, std_val in enumerate(channel_stds):
        if std_val == 0.0:
            channel_stds[i] = 1.0
            # Log a warning so the user knows this channel is constant
            warnings.warn(
                f"Channel {i} has zero standard deviation in training data. "
                f"Setting std to 1.0 to avoid division by zero.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Channel %d has zero standard deviation. Setting std=1.0.", i
            )

    # Return statistics as a dictionary for easy JSON serialization
    stats: dict[str, list[float]] = {"mean": channel_means, "std": channel_stds}
    return stats


def normalize(data: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    """Apply z-score normalization using precomputed mean and std statistics.

    Each channel is independently normalized: (value - mean) / std.

    Parameters:
        data: A numpy array of shape (time_steps, num_channels) or (time_steps,)
              containing raw time series values.
        stats: A dictionary with "mean" and "std" keys, each containing a list
               of per-channel float values (as returned by compute_normalization_stats).

    Returns:
        A numpy array of the same shape as the input, with z-score normalized values.
    """
    # Handle 1D input by temporarily reshaping to 2D
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(-1, 1)

    # Convert stats lists to numpy arrays for vectorized operations
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)

    # Apply z-score normalization: subtract mean, divide by std
    normalized = (data - mean) / std

    # Restore original shape if input was 1D
    if was_1d:
        normalized = normalized.ravel()

    return normalized


def inverse_normalize(data: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    """Reverse z-score normalization to recover values in the original data scale.

    Each channel is independently de-normalized: value * std + mean.

    Parameters:
        data: A numpy array of shape (time_steps, num_channels) or (time_steps,)
              containing normalized time series values.
        stats: A dictionary with "mean" and "std" keys, each containing a list
               of per-channel float values (as returned by compute_normalization_stats).

    Returns:
        A numpy array of the same shape as the input, with values in the original scale.
    """
    # Handle 1D input by temporarily reshaping to 2D
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(-1, 1)

    # Convert stats lists to numpy arrays for vectorized operations
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)

    # Reverse the z-score transformation: multiply by std, then add mean
    original_scale = data * std + mean

    # Restore original shape if input was 1D
    if was_1d:
        original_scale = original_scale.ravel()

    return original_scale


def split_chronological(
    data: np.ndarray,
    train_ratio: float = Config.TRAIN_RATIO,
    val_ratio: float = Config.VAL_RATIO,
    test_ratio: float = Config.TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split time series data into train/val/test sets preserving chronological order.

    The split is performed along the first axis (time axis) without any shuffling,
    ensuring that the training set contains the earliest data, validation is in the
    middle, and test is the most recent data. This prevents future data leakage.

    Parameters:
        data: A numpy array of shape (time_steps, ...) to be split. Can be 1D
              (univariate) or 2D (multivariate with channels).
        train_ratio: Fraction of data for training (default 0.70).
        val_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).

    Returns:
        A tuple of three numpy arrays (train, val, test) that are contiguous,
        non-overlapping slices of the original data along the time axis.
    """
    # Total number of time steps in the dataset
    n = len(data)

    # Compute split indices based on ratios (integer boundaries)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Slice the data chronologically — no shuffling to preserve time order
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


def filter_short_series(
    series_list: list[np.ndarray],
    min_length: int = Config.CONTEXT_LENGTH,
    series_ids: list[str] | None = None,
) -> list[np.ndarray]:
    """Filter out time series that are shorter than the minimum required length.

    Series shorter than one context window (512 time steps by default) cannot
    be used for training or evaluation, so they are discarded with a warning.

    Parameters:
        series_list: A list of numpy arrays, each representing a time series.
                     Arrays can be 1D (univariate) or 2D (time_steps, channels).
        min_length: Minimum number of time steps required (default 512 from Config).
        series_ids: Optional list of identifiers for each series (for logging).
                    If None, integer indices are used in warning messages.

    Returns:
        A list containing only the series with length >= min_length, preserving
        the original order of retained series.
    """
    # Build list of valid series, logging warnings for discarded ones
    valid_series: list[np.ndarray] = []

    for i, series in enumerate(series_list):
        # Determine the length along the time axis (first dimension)
        series_length = len(series)

        if series_length < min_length:
            # Identify the series for the warning message
            series_id = series_ids[i] if series_ids is not None else str(i)
            # Warn the user that this series is being discarded
            warnings.warn(
                f"Series '{series_id}' has length {series_length} which is shorter "
                f"than the minimum required length of {min_length} time steps. "
                f"Discarding this series.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Discarding series '%s' with length %d (minimum: %d).",
                series_id,
                series_length,
                min_length,
            )
        else:
            # Series is long enough — keep it
            valid_series.append(series)

    return valid_series


def save_normalization_stats(
    stats: dict[str, Any],
    dataset_name: str,
    output_dir: str = "data/processed",
) -> str:
    """Save normalization statistics to a JSON file for later inverse transformation.

    The stats are saved with the dataset name as a key, allowing multiple datasets'
    statistics to be stored in the same file or in separate files.

    Parameters:
        stats: A dictionary with "mean" and "std" keys (from compute_normalization_stats).
        dataset_name: Name of the dataset (e.g., "energy", "weather", "finance", "etth1").
        output_dir: Directory where the JSON file will be saved (default: data/processed/).

    Returns:
        The file path where the statistics were saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    filepath = os.path.join(output_dir, f"{dataset_name}_norm_stats.json")

    # Wrap stats under the dataset name key for clarity
    output_data = {dataset_name: stats}

    # Write the statistics as formatted JSON for human readability
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved normalization stats for '%s' to %s", dataset_name, filepath)
    return filepath


def load_normalization_stats(
    dataset_name: str,
    input_dir: str = "data/processed",
) -> dict[str, list[float]]:
    """Load normalization statistics from a previously saved JSON file.

    Parameters:
        dataset_name: Name of the dataset (e.g., "energy", "weather", "finance", "etth1").
        input_dir: Directory where the JSON file is stored (default: data/processed/).

    Returns:
        A dictionary with "mean" and "std" keys containing per-channel float lists.

    Raises:
        FileNotFoundError: If the stats file does not exist.
    """
    # Construct the expected file path
    filepath = os.path.join(input_dir, f"{dataset_name}_norm_stats.json")

    # Read and parse the JSON file
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the stats for the requested dataset
    return data[dataset_name]


def preprocess_dataset(
    raw_data: np.ndarray,
    dataset_name: str,
    output_dir: str = "data/processed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Full preprocessing pipeline: split, compute stats, normalize, and save.

    This is a convenience function that chains together the individual preprocessing
    steps in the correct order:
    1. Split data chronologically into train/val/test
    2. Compute normalization statistics from training split only
    3. Normalize all three splits using training statistics
    4. Save normalization statistics as JSON

    Parameters:
        raw_data: A numpy array of shape (time_steps, num_channels) or (time_steps,)
                  containing the raw time series data.
        dataset_name: Name of the dataset for saving stats (e.g., "energy").
        output_dir: Directory for saving normalization stats JSON.

    Returns:
        A tuple of (train_normalized, val_normalized, test_normalized, stats) where
        each split is a normalized numpy array and stats is the normalization dictionary.
    """
    # Step 1: Split the data chronologically (70/15/15)
    train, val, test = split_chronological(raw_data)

    # Step 2: Compute normalization statistics from training data only
    # This prevents information leakage from validation/test into training
    stats = compute_normalization_stats(train)

    # Step 3: Normalize all splits using the training statistics
    train_normalized = normalize(train, stats)
    val_normalized = normalize(val, stats)
    test_normalized = normalize(test, stats)

    # Step 4: Save the normalization statistics for later inverse transformation
    save_normalization_stats(stats, dataset_name, output_dir)

    return train_normalized, val_normalized, test_normalized, stats
