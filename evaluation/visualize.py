"""Visualization module for probabilistic forecast evaluation.

This module generates plots comparing actual values against model predictions
with shaded prediction intervals (P10-P90). It selects test windows evenly
spaced across the ETTh1 test split, renders them with descriptive titles,
axis labels, and legends, and saves the output as PNG files at 150+ DPI.
Plots are also displayed inline when running in Google Colab.

Related modules:
    - forecasting/inference.py provides zero_shot_forecast and compute_num_windows
      for generating probabilistic predictions on the test split.
    - data/preprocess.py provides inverse_normalize for converting normalized
      targets back to the original data scale.
    - config.py supplies CONTEXT_LENGTH, FORECAST_HORIZON, and QUANTILES.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import Config


def _is_colab_environment() -> bool:
    """Detect whether the code is running inside Google Colab.

    Returns:
        True if running in Colab, False otherwise.
    """
    try:
        # google.colab is only available inside Colab notebooks
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _configure_matplotlib_backend() -> None:
    """Configure matplotlib backend for the current environment.

    In Colab, uses the default inline backend so plots render in notebook cells.
    In non-interactive environments, uses the 'Agg' backend for file-only output.
    """
    if _is_colab_environment():
        # Colab uses inline rendering by default — no change needed
        pass
    else:
        # Use non-interactive backend for saving to files without a display
        matplotlib.use("Agg")


def select_window_indices(
    num_windows: int,
    num_plots: int = 5,
) -> list[int]:
    """Select window indices evenly spaced across the available test windows.

    This ensures the visualized windows cover the full temporal extent of the
    test split rather than clustering at the beginning or end.

    Parameters:
        num_windows: Total number of available forecast windows in the test split.
        num_plots: Number of windows to select for plotting (default 5).

    Returns:
        A list of integer indices into the forecast windows array, evenly spaced
        from the first to the last window. If num_windows <= num_plots, returns
        all available indices.
    """
    # If fewer windows than requested plots, just use all of them
    if num_windows <= num_plots:
        return list(range(num_windows))

    # Compute evenly spaced indices from 0 to num_windows - 1
    indices = np.linspace(0, num_windows - 1, num_plots, dtype=int).tolist()

    return indices


def plot_forecast_window(
    actual: np.ndarray,
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    window_index: int,
    output_dir: str = "evaluation",
    dpi: int = 150,
    show_inline: bool = True,
) -> str:
    """Plot a single forecast window with actual values and prediction intervals.

    Generates a figure showing:
    - Actual values as a solid blue line
    - P50 (median) predictions as a dashed orange line
    - Shaded region between P10 and P90 representing the 80% prediction interval

    Parameters:
        actual: 1D numpy array of actual values for the forecast horizon (96 steps).
        p50: 1D numpy array of P50 (median) predictions (96 steps).
        p10: 1D numpy array of P10 (10th percentile) predictions (96 steps).
        p90: 1D numpy array of P90 (90th percentile) predictions (96 steps).
        window_index: Integer index of this window in the test split (for title).
        output_dir: Directory where the PNG file will be saved (default: evaluation/).
        dpi: Resolution of the saved PNG file (default 150, minimum required).
        show_inline: Whether to display the plot inline (True in Colab).

    Returns:
        The file path where the PNG was saved.
    """
    # Create a time axis for the forecast horizon (0 to 95 for 96 steps)
    time_steps = np.arange(len(actual))

    # Create figure with a reasonable size for readability
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot actual values as a solid blue line
    ax.plot(
        time_steps,
        actual,
        color="blue",
        linewidth=1.5,
        label="Actual",
        zorder=3,
    )

    # Plot P50 (median) predictions as a dashed orange line
    ax.plot(
        time_steps,
        p50,
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
        label="P50 Prediction",
        zorder=3,
    )

    # Shade the P10-P90 prediction interval in light orange
    ax.fill_between(
        time_steps,
        p10,
        p90,
        color="orange",
        alpha=0.25,
        label="P10-P90 Interval",
        zorder=2,
    )

    # Add descriptive title indicating which test window this is
    ax.set_title(
        f"Forecast Window {window_index} — ETTh1 Test Split "
        f"(Horizon: {Config.FORECAST_HORIZON} steps)",
        fontsize=12,
    )

    # Label axes clearly for readability
    ax.set_xlabel("Time Step (within forecast horizon)", fontsize=10)
    ax.set_ylabel("Value (original scale)", fontsize=10)

    # Add legend to distinguish the three visual elements
    ax.legend(loc="upper right", fontsize=9)

    # Add a light grid for easier value reading
    ax.grid(True, alpha=0.3, linestyle="-")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path with window index in the filename
    filename = f"forecast_window_{window_index}.png"
    filepath = os.path.join(output_dir, filename)

    # Save the figure as PNG at the specified DPI (minimum 150 required)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    # Display inline if running in Colab
    if show_inline:
        plt.show()
    else:
        # Close the figure to free memory when not displaying
        plt.close(fig)

    return filepath


def visualize_forecasts(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    num_plots: int = 5,
    output_dir: str = "evaluation",
    dpi: int = 150,
) -> list[str]:
    """Generate visualization plots for multiple forecast windows.

    Selects windows evenly spaced across the test split and generates a plot
    for each one showing actual values, P50 predictions, and P10-P90 intervals.
    Plots are saved as PNG files and displayed inline in Colab.

    Parameters:
        actuals: 2D numpy array of shape (num_windows, forecast_horizon) containing
                 the actual target values for each test window in original scale.
        forecasts: 3D numpy array of shape (num_windows, forecast_horizon, 3)
                   containing P10/P50/P90 quantile forecasts in original scale.
                   Channel order: [P10, P50, P90] along axis 2.
        num_plots: Number of windows to visualize (default 5, evenly spaced).
        output_dir: Directory where PNG files will be saved (default: evaluation/).
        dpi: Resolution of saved PNG files (default 150, minimum required).

    Returns:
        A list of file paths to the saved PNG files.
    """
    # Configure matplotlib backend based on the runtime environment
    _configure_matplotlib_backend()

    # Determine whether to show plots inline (only in Colab)
    show_inline = _is_colab_environment()

    # Total number of available forecast windows
    num_windows = actuals.shape[0]

    # Select evenly spaced window indices for visualization
    selected_indices = select_window_indices(num_windows, num_plots)

    # Store paths to all saved plot files
    saved_paths: list[str] = []

    # Generate a plot for each selected window
    for window_idx in selected_indices:
        # Extract actual values for this window (shape: forecast_horizon)
        actual = actuals[window_idx]

        # Extract quantile predictions for this window
        # forecasts shape: (num_windows, forecast_horizon, 3) → [P10, P50, P90]
        p10 = forecasts[window_idx, :, 0]
        p50 = forecasts[window_idx, :, 1]
        p90 = forecasts[window_idx, :, 2]

        # Generate and save the plot for this window
        filepath = plot_forecast_window(
            actual=actual,
            p50=p50,
            p10=p10,
            p90=p90,
            window_index=window_idx,
            output_dir=output_dir,
            dpi=dpi,
            show_inline=show_inline,
        )

        saved_paths.append(filepath)

    # Print summary of saved plots
    print(f"\nSaved {len(saved_paths)} forecast visualization plots to '{output_dir}/':")
    for path in saved_paths:
        print(f"  - {path}")

    return saved_paths


def visualize_from_data(
    test_data: np.ndarray,
    forecasts: np.ndarray,
    norm_stats: dict[str, list[float]],
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
    num_plots: int = 5,
    output_dir: str = "evaluation",
    dpi: int = 150,
) -> list[str]:
    """High-level visualization function that extracts actuals from test data.

    This convenience function takes the normalized test data and forecast output,
    extracts the actual target values for each window, applies inverse normalization,
    and generates the visualization plots.

    Parameters:
        test_data: 1D numpy array of normalized test split values.
        forecasts: 3D numpy array of shape (num_windows, forecast_horizon, 3)
                   containing P10/P50/P90 forecasts in original scale.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of forecast steps per window (default 96).
        stride: Step size between consecutive windows (default 96).
        num_plots: Number of windows to visualize (default 5).
        output_dir: Directory where PNG files will be saved.
        dpi: Resolution of saved PNG files (minimum 150).

    Returns:
        A list of file paths to the saved PNG files.
    """
    # Import inverse_normalize here to avoid circular imports at module level
    from data.preprocess import inverse_normalize

    # Compute the number of forecast windows in the test data
    num_windows = forecasts.shape[0]

    # Extract actual target values for each window and inverse-normalize them
    actuals_list: list[np.ndarray] = []

    for window_idx in range(num_windows):
        # The target starts right after the context window
        target_start = window_idx * stride + context_length
        target_end = target_start + forecast_horizon

        # Extract the normalized actual values for this window
        actual_normalized = test_data[target_start:target_end]

        # Apply inverse normalization to get values in original scale
        actual_original = inverse_normalize(actual_normalized, norm_stats)

        actuals_list.append(actual_original)

    # Stack all actual windows into a 2D array: (num_windows, forecast_horizon)
    actuals = np.stack(actuals_list, axis=0)

    # Generate the visualization plots
    saved_paths = visualize_forecasts(
        actuals=actuals,
        forecasts=forecasts,
        num_plots=num_plots,
        output_dir=output_dir,
        dpi=dpi,
    )

    return saved_paths
