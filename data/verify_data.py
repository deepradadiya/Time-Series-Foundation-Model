"""Data verification script for the Time Series Foundation Model.

This module loads all processed datasets, computes summary statistics,
prints a formatted table, and plots sample time series for visual inspection.
It serves as a quick sanity check after running the preprocessing pipeline.

Related modules:
    - data/preprocess_pipeline.py: Produces the .pt files this script verifies
    - data/patching.py: Provides compute_num_patches for patch count stats
    - config.py: Provides CONTEXT_LENGTH, PATCH_LEN, PATCH_STRIDE
"""

import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.patching import compute_num_patches


# Datasets to verify (order matters for display)
DATASETS = ["energy", "weather", "finance", "etth1"]
SPLITS = ["train", "val", "test"]


def load_processed_dataset(
    dataset_name: str, processed_dir: str = "data/processed"
) -> dict[str, dict[str, torch.Tensor]]:
    """Load all three split .pt files for a processed dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "energy", "weather").
        processed_dir: Directory containing processed .pt files.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing
        a dict with 'context' and 'target' tensors.

    Raises:
        FileNotFoundError: If any split file is missing.
        RuntimeError: If any split file is corrupt or cannot be loaded.
    """
    splits = {}
    for split in SPLITS:
        filepath = os.path.join(processed_dir, f"{dataset_name}_{split}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing processed file: {filepath}"
            )
        splits[split] = torch.load(filepath, weights_only=False)
    return splits


def compute_dataset_stats(dataset_name: str, splits: dict[str, dict[str, torch.Tensor]]) -> dict:
    """Compute summary statistics for a processed dataset.

    Args:
        dataset_name: Name of the dataset.
        splits: Dictionary with keys 'train', 'val', 'test', each containing
            a dict with 'context' and 'target' tensors.

    Returns:
        Dictionary with keys:
            - train_samples: number of training samples
            - val_samples: number of validation samples
            - test_samples: number of test samples
            - num_patches: number of patches per context window
            - value_min: minimum value across all splits
            - value_max: maximum value across all splits
    """
    train_samples = splits["train"]["context"].shape[0]
    val_samples = splits["val"]["context"].shape[0]
    test_samples = splits["test"]["context"].shape[0]

    # Compute num_patches from context length
    num_patches = compute_num_patches(
        Config.CONTEXT_LENGTH, Config.PATCH_LEN, Config.PATCH_STRIDE
    )

    # Compute value range across all splits (context and target)
    all_values = []
    for split in SPLITS:
        context = splits[split]["context"]
        target = splits[split]["target"]
        if context.numel() > 0:
            all_values.append(context)
        if target.numel() > 0:
            all_values.append(target)

    if all_values:
        combined = torch.cat([v.flatten() for v in all_values])
        value_min = combined.min().item()
        value_max = combined.max().item()
    else:
        value_min = float("nan")
        value_max = float("nan")

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "num_patches": num_patches,
        "value_min": value_min,
        "value_max": value_max,
    }


def print_summary_table(all_stats: dict[str, dict]) -> None:
    """Print formatted summary table of all datasets.

    Columns: Dataset, Train samples, Val samples, Test samples, Num patches, Value range.
    ETTh1's Num patches column is annotated with "[ZERO-SHOT ONLY]".

    Args:
        all_stats: Dictionary mapping dataset names to their stats dicts.
    """
    # Header
    header = (
        f"{'Dataset':<12} {'Train samples':>14} {'Val samples':>12} "
        f"{'Test samples':>13} {'Num patches':>20} {'Value range':>20}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for dataset_name, stats in all_stats.items():
        # Format num_patches with annotation for ETTh1
        if dataset_name == "etth1":
            patches_str = f"{stats['num_patches']} [ZERO-SHOT ONLY]"
        else:
            patches_str = str(stats["num_patches"])

        # Format value range
        value_range = f"[{stats['value_min']:.4f}, {stats['value_max']:.4f}]"

        row = (
            f"{dataset_name:<12} {stats['train_samples']:>14} {stats['val_samples']:>12} "
            f"{stats['test_samples']:>13} {patches_str:>20} {value_range:>20}"
        )
        print(row)

    print(separator)


def plot_sample_series(
    all_splits: dict[str, dict[str, dict[str, torch.Tensor]]],
    output_path: str | None = None,
) -> None:
    """Plot first test context window (512 steps) from each dataset as line charts.

    Args:
        all_splits: Dictionary mapping dataset names to their splits dicts.
        output_path: Optional path to save the figure. If None, displays interactively.
    """
    # Filter datasets that have test samples
    plottable = {
        name: splits
        for name, splits in all_splits.items()
        if splits["test"]["context"].shape[0] > 0
    }

    if not plottable:
        print("No datasets have test samples to plot.")
        return

    num_plots = len(plottable)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), squeeze=False)

    for idx, (dataset_name, splits) in enumerate(plottable.items()):
        ax = axes[idx, 0]
        # Get first test context window (512 time steps)
        context = splits["test"]["context"][0].numpy()
        ax.plot(context, linewidth=0.8)
        ax.set_title(f"{dataset_name} - First test context window ({Config.CONTEXT_LENGTH} steps)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value (normalized)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.savefig("data/processed/sample_series.png", dpi=100, bbox_inches="tight")
        print("Plot saved to: data/processed/sample_series.png")

    plt.close(fig)


def verify_all(processed_dir: str = "data/processed") -> None:
    """Main verification entry point.

    Loads all processed datasets, computes statistics, prints a summary table,
    and plots sample time series. Handles missing or corrupt files gracefully
    by printing an error, skipping the dataset, and continuing.

    Args:
        processed_dir: Directory containing processed .pt files.
    """
    all_stats = {}
    all_splits = {}

    for dataset_name in DATASETS:
        try:
            splits = load_processed_dataset(dataset_name, processed_dir)
            stats = compute_dataset_stats(dataset_name, splits)
            all_stats[dataset_name] = stats
            all_splits[dataset_name] = splits
        except FileNotFoundError as e:
            print(f"ERROR: {e} — skipping '{dataset_name}'")
            continue
        except Exception as e:
            print(f"ERROR: Failed to load '{dataset_name}': {e} — skipping")
            continue

    if all_stats:
        print("\n=== Data Pipeline Verification Summary ===\n")
        print_summary_table(all_stats)
        print()

    if all_splits:
        plot_sample_series(all_splits)


if __name__ == "__main__":
    verify_all()
