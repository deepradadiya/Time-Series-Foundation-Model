"""Preprocessing pipeline for the Time Series Foundation Model.

This module orchestrates the full preprocessing pipeline for all datasets.
It loads raw CSV files, applies chronological splitting, z-score normalization
(using training statistics only), sliding window extraction, and saves the
results as PyTorch tensors for efficient training and evaluation.

Related modules:
    - data/preprocess.py: Provides split_chronological, compute_normalization_stats,
      normalize, and save_normalization_stats functions.
    - config.py: Provides CONTEXT_LENGTH (512), FORECAST_HORIZON (96), and split ratios.
    - data/patching.py: Provides create_patches for downstream use.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

from config import Config
from data import preprocess

# Set up module-level logger
logger = logging.getLogger(__name__)


def create_windows(
    data: np.ndarray,
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sliding windows from a 1-D normalized time series.

    Each window consists of a context segment (input) and a target segment
    (forecast label) that immediately follows the context in the original series.

    Args:
        data: 1-D array of normalized time steps.
        context_length: Input window size (default 512).
        forecast_horizon: Target window size (default 96).
        stride: Step between consecutive windows (default 96).

    Returns:
        Tuple of (contexts, targets) arrays:
          - contexts: shape (num_samples, context_length)
          - targets: shape (num_samples, forecast_horizon)
        If the series is too short (< context_length + forecast_horizon),
        returns empty arrays with shapes (0, context_length) and (0, forecast_horizon).
    """
    window_size = context_length + forecast_horizon

    # Handle series too short for even one window
    if len(data) < window_size:
        return (
            np.empty((0, context_length), dtype=data.dtype),
            np.empty((0, forecast_horizon), dtype=data.dtype),
        )

    # Compute number of windows: (L - window_size) // stride + 1
    num_samples = (len(data) - window_size) // stride + 1

    # Pre-allocate output arrays
    contexts = np.empty((num_samples, context_length), dtype=data.dtype)
    targets = np.empty((num_samples, forecast_horizon), dtype=data.dtype)

    # Extract each window
    for i in range(num_samples):
        start = i * stride
        contexts[i] = data[start : start + context_length]
        targets[i] = data[start + context_length : start + window_size]

    return contexts, targets


def process_dataset(
    dataset_name: str,
    raw_path: str,
    output_dir: str = "data/processed",
) -> dict[str, int]:
    """Full preprocessing for a single dataset.

    Steps:
      1. Load CSV, extract 'value' column as numpy array
      2. Chronological split (70/15/15) via preprocess.split_chronological
      3. Compute normalization stats from train split only
      4. Normalize all splits via preprocess.normalize
      5. Extract sliding windows from each split
      6. Save as PyTorch .pt tensors
      7. Save normalization stats as JSON

    Args:
        dataset_name: Name of the dataset (e.g., "energy", "weather").
        raw_path: Path to the raw CSV file with a "value" column.
        output_dir: Directory for saving processed .pt files and stats JSON.

    Returns:
        Dictionary with sample counts: {'train': N, 'val': N, 'test': N}

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If the CSV does not contain a 'value' column.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load CSV and extract "value" column as numpy array
    df = pd.read_csv(raw_path)
    if "value" not in df.columns:
        raise ValueError(
            f"CSV file '{raw_path}' does not contain a 'value' column. "
            f"Available columns: {list(df.columns)}"
        )
    values = df["value"].to_numpy(dtype=np.float64)

    # Step 2: Chronological split (70/15/15)
    train_data, val_data, test_data = preprocess.split_chronological(values)

    # Step 3: Compute normalization stats from train split only
    stats = preprocess.compute_normalization_stats(train_data)

    # Step 4: Normalize all three splits using training statistics
    train_norm = preprocess.normalize(train_data, stats)
    val_norm = preprocess.normalize(val_data, stats)
    test_norm = preprocess.normalize(test_data, stats)

    # Step 5: Extract sliding windows from each normalized split
    splits = {"train": train_norm, "val": val_norm, "test": test_norm}
    sample_counts = {}

    for split_name, split_data in splits.items():
        contexts, targets = create_windows(split_data)

        num_samples = contexts.shape[0]
        sample_counts[split_name] = num_samples

        # Log warning if split is too short for any windows
        if num_samples == 0:
            logger.warning(
                "Dataset '%s' split '%s' has only %d time steps, which is fewer "
                "than the minimum %d required for one window. Saving empty tensors.",
                dataset_name,
                split_name,
                len(split_data),
                Config.CONTEXT_LENGTH + Config.FORECAST_HORIZON,
            )

        # Step 6: Save as PyTorch .pt file
        tensor_dict = {
            "context": torch.tensor(contexts, dtype=torch.float32),
            "target": torch.tensor(targets, dtype=torch.float32),
        }
        output_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.pt")
        torch.save(tensor_dict, output_path)
        logger.info(
            "Saved %s_%s.pt: %d samples (context: %s, target: %s)",
            dataset_name,
            split_name,
            num_samples,
            tensor_dict["context"].shape,
            tensor_dict["target"].shape,
        )

    # Step 7: Save normalization stats as JSON
    preprocess.save_normalization_stats(stats, dataset_name, output_dir)

    return sample_counts


def run_pipeline(datasets: list[str] | None = None) -> None:
    """Run the full preprocessing pipeline for specified datasets.

    Args:
        datasets: List of dataset names to process.
                  Defaults to ['energy', 'weather', 'finance', 'etth1'].
    """
    if datasets is None:
        datasets = ["energy", "weather", "finance", "etth1"]

    for dataset_name in datasets:
        raw_path = os.path.join("data", "raw", f"{dataset_name}.csv")

        if not os.path.exists(raw_path):
            logger.error(
                "Raw file not found for dataset '%s': %s. "
                "Please run the download script first.",
                dataset_name,
                raw_path,
            )
            continue

        logger.info("Processing dataset: %s", dataset_name)
        try:
            counts = process_dataset(dataset_name, raw_path)
            logger.info(
                "Dataset '%s' processed successfully. "
                "Samples — train: %d, val: %d, test: %d",
                dataset_name,
                counts["train"],
                counts["val"],
                counts["test"],
            )
        except Exception as e:
            logger.error(
                "Failed to process dataset '%s': %s", dataset_name, e
            )
            raise


if __name__ == "__main__":
    # Configure logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_pipeline()
