"""Publication-quality visualization module for the evaluation pipeline.

This module generates three types of plots:
1. Forecast plot — actual vs predicted with P10-P90 uncertainty bands
2. Pretraining loss curve — per-domain reconstruction loss over epochs
3. MAE comparison bar chart — side-by-side MAE for all 5 models

All plots use the non-interactive matplotlib 'Agg' backend for file-only output
and are saved as PNG files at a minimum of 300 DPI.

Related modules:
    - evaluation/metrics.py provides metric computations
    - evaluation/results_table.py provides formatted results
    - forecasting/zero_shot_eval.py and forecasting/finetune_eval.py produce forecasts
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file-only output
import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(
    actual: np.ndarray,
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    window_index: int,
    dataset_name: str = "ETTh1",
    output_dir: str = "evaluation/results",
    dpi: int = 300,
) -> str:
    """Generate a forecast plot: actual (solid) vs P50 (dashed) with P10-P90 shading.

    Plots the actual time series as a solid line and the P50 median prediction as a
    dashed line, with the P10-P90 prediction interval shaded to show uncertainty.

    Parameters:
        actual: 1D numpy array of actual values for the forecast horizon.
        p50: 1D numpy array of P50 (median) predictions.
        p10: 1D numpy array of P10 (10th percentile) predictions.
        p90: 1D numpy array of P90 (90th percentile) predictions.
        window_index: Integer index of this forecast window (used in title/filename).
        dataset_name: Name of the dataset (default "ETTh1", used in title).
        output_dir: Directory where the PNG file will be saved.
        dpi: Resolution of the saved PNG file (minimum 300).

    Returns:
        The file path where the PNG was saved.

    Raises:
        ValueError: If actual, p50, p10, and p90 arrays have different lengths.
    """
    # Convert to numpy arrays if not already
    actual = np.asarray(actual)
    p50 = np.asarray(p50)
    p10 = np.asarray(p10)
    p90 = np.asarray(p90)

    # Validate that all arrays have the same length
    lengths = [len(actual), len(p50), len(p10), len(p90)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Array length mismatch: actual={len(actual)}, p50={len(p50)}, "
            f"p10={len(p10)}, p90={len(p90)}. All arrays must have the same length."
        )

    # Create time axis
    time_steps = np.arange(len(actual))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot actual values as a solid line
    ax.plot(
        time_steps,
        actual,
        color="blue",
        linewidth=1.5,
        linestyle="-",
        label="Actual",
        zorder=3,
    )

    # Plot P50 predictions as a dashed line
    ax.plot(
        time_steps,
        p50,
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
        label="P50 Prediction",
        zorder=3,
    )

    # Shade the P10-P90 prediction interval
    ax.fill_between(
        time_steps,
        p10,
        p90,
        color="orange",
        alpha=0.3,
        label="P10-P90 Interval",
        zorder=2,
    )

    # Title with window index and dataset name
    ax.set_title(
        f"Forecast Window {window_index} — {dataset_name}",
        fontsize=12,
    )

    # Axis labels
    ax.set_xlabel("Time Steps", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    # Legend
    ax.legend(loc="upper right", fontsize=9)

    # Light grid for readability
    ax.grid(True, alpha=0.3, linestyle="-")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    filename = f"forecast_window_{window_index}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_loss_curve(
    domain_losses: dict[str, list[float]],
    output_dir: str = "evaluation/results",
    dpi: int = 300,
) -> str:
    """Plot pretraining loss curves for Energy, Weather, Finance domains.

    Displays reconstruction loss on the y-axis and epoch number on the x-axis,
    with one line per domain. Domains with zero epochs of data are omitted.

    Parameters:
        domain_losses: Dictionary mapping domain names (e.g., "Energy", "Weather",
                       "Finance") to lists of per-epoch loss values.
        output_dir: Directory where the PNG file will be saved.
        dpi: Resolution of the saved PNG file (minimum 300).

    Returns:
        The file path where the PNG was saved.
    """
    # Define distinct colors for each domain
    domain_colors = {
        "Energy": "#1f77b4",    # Blue
        "Weather": "#2ca02c",   # Green
        "Finance": "#d62728",   # Red
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each domain that has data
    has_data = False
    for domain_name, losses in domain_losses.items():
        if losses and len(losses) > 0:
            epochs = list(range(1, len(losses) + 1))
            color = domain_colors.get(domain_name, None)
            ax.plot(
                epochs,
                losses,
                linewidth=1.5,
                label=domain_name,
                color=color,
                marker="o",
                markersize=4,
            )
            has_data = True

    # Configure axes and labels
    ax.set_title("Pretraining Loss Curve", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Reconstruction Loss", fontsize=10)

    # Add legend only if there's data to show
    if has_data:
        ax.legend(loc="upper right", fontsize=9)

    # Light grid
    ax.grid(True, alpha=0.3, linestyle="-")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    filepath = os.path.join(output_dir, "pretraining_loss_curve.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_mae_bar_chart(
    model_maes: dict[str, float],
    output_dir: str = "evaluation/results",
    dpi: int = 300,
) -> str:
    """Bar chart comparing MAE across all 5 models.

    Displays one bar per model in the specified order, with baseline models in one
    color and PatchTST models in another. Value labels are placed on top of each bar.

    Parameters:
        model_maes: Dictionary mapping model names to their MAE values.
                    Expected keys: "Naive", "ARIMA", "Prophet",
                    "PatchTST zero-shot", "PatchTST fine-tuned".
        output_dir: Directory where the PNG file will be saved.
        dpi: Resolution of the saved PNG file (minimum 300).

    Returns:
        The file path where the PNG was saved.
    """
    # Define the fixed order of models
    model_order = [
        "Naive",
        "ARIMA",
        "Prophet",
        "PatchTST zero-shot",
        "PatchTST fine-tuned",
    ]

    # Separate baseline and PatchTST models for coloring
    baseline_models = {"Naive", "ARIMA", "Prophet"}
    baseline_color = "#1f77b4"   # Blue for baselines
    patchtst_color = "#ff7f0e"   # Orange for PatchTST models

    # Build ordered lists of names, values, and colors
    names = []
    values = []
    colors = []
    for model_name in model_order:
        if model_name in model_maes:
            names.append(model_name)
            values.append(model_maes[model_name])
            if model_name in baseline_models:
                colors.append(baseline_color)
            else:
                colors.append(patchtst_color)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x_positions = np.arange(len(names))
    bars = ax.bar(x_positions, values, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Add value labels on top of each bar (4 decimal places)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("MAE Comparison", fontsize=12)

    # Y-axis starts at zero
    ax.set_ylim(bottom=0)

    # Add legend to distinguish baseline vs PatchTST
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=baseline_color, edgecolor="black", linewidth=0.5, label="Baselines"),
        Patch(facecolor=patchtst_color, edgecolor="black", linewidth=0.5, label="PatchTST"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Light grid on y-axis only
    ax.grid(True, alpha=0.3, linestyle="-", axis="y")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    filepath = os.path.join(output_dir, "mae_comparison_bar_chart.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath
