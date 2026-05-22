"""Interactive Gradio forecasting demo for the Time Series Foundation Model.

This module implements the HuggingFace Space entry point that allows users to
upload a CSV file, select a target column, choose a forecast horizon, and receive
probabilistic forecasts (P10/P50/P90) displayed as an interactive plot. The app
loads a pretrained PatchTST model and runs inference on the last 512 time steps
of the selected column.

Related modules:
    - model/patchtst.py provides the PatchTSTModel encoder backbone.
    - forecasting/probabilistic_head.py provides the ProbabilisticForecastHead.
    - data/preprocess.py provides normalize and compute_normalization_stats.
    - config.py supplies CONTEXT_LENGTH (512), FORECAST_HORIZON (96), and other
      hyperparameters used during inference.
"""

import os
import sys
import time
from typing import Optional

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path for imports when running as HuggingFace Space entry point
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from data.preprocess import compute_normalization_stats, normalize, inverse_normalize
from model.patchtst import PatchTSTModel
from forecasting.probabilistic_head import ProbabilisticForecastHead

# Use non-interactive matplotlib backend for server-side rendering
matplotlib.use("Agg")

# App constants
MAX_FILE_SIZE_MB = 50
MIN_ROWS = Config.CONTEXT_LENGTH  # 512 rows minimum
INFERENCE_TIMEOUT_SECONDS = 30
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pretrained_model.pt")

# Global model cache to avoid reloading on every forecast request
_cached_model: Optional[PatchTSTModel] = None
_cached_head: Optional[ProbabilisticForecastHead] = None


def load_model() -> tuple[PatchTSTModel, ProbabilisticForecastHead]:
    """Load the pretrained PatchTST model and probabilistic forecast head.

    Uses a global cache so the model is only loaded once per session. If no
    pretrained checkpoint is found, initializes with random weights.

    Returns:
        A tuple of (PatchTSTModel, ProbabilisticForecastHead) ready for inference.
    """
    global _cached_model, _cached_head

    # Return cached model if already loaded
    if _cached_model is not None and _cached_head is not None:
        return _cached_model, _cached_head

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate model and forecast head with default configuration
    model = PatchTSTModel(Config)
    head = ProbabilisticForecastHead(
        d_model=Config.D_MODEL, num_patches=Config.NUM_PATCHES,
        forecast_horizon=Config.FORECAST_HORIZON, quantiles=Config.QUANTILES,
    )

    # Attempt to load pretrained weights from checkpoint
    checkpoint_path = _find_checkpoint()
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "head_state_dict" in checkpoint:
            head.load_state_dict(checkpoint["head_state_dict"])

    # Move to device and set to evaluation mode (disables dropout)
    model = model.to(device).eval()
    head = head.to(device).eval()

    # Cache for subsequent calls
    _cached_model, _cached_head = model, head
    return model, head


def _find_checkpoint() -> Optional[str]:
    """Find the most recent checkpoint file in the checkpoints directory.

    Returns:
        Path to the most recent checkpoint file, or None if not found.
    """
    if os.path.isfile(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    # Search for .pt files and return the most recently modified one
    pt_files = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")
    ]
    return max(pt_files, key=os.path.getmtime) if pt_files else None


def validate_csv(file_path: str) -> tuple[bool, str, Optional[pd.DataFrame]]:
    """Validate an uploaded CSV file for forecasting requirements.

    Checks: parseable CSV, >= 512 rows, has datetime column, has numeric column.

    Parameters:
        file_path: Path to the uploaded CSV file.

    Returns:
        A tuple of (is_valid, error_message, dataframe).
    """
    # Attempt to read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return False, f"Failed to parse CSV file: {str(e)}", None

    # Check minimum row count
    if len(df) < MIN_ROWS:
        return (False,
                f"CSV has {len(df)} rows, but the minimum required is {MIN_ROWS} "
                f"(Context Window length). Please upload a file with at least "
                f"{MIN_ROWS} data rows.", None)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return (False,
                "No numeric columns found in the CSV. The file must contain at "
                "least one numeric (int or float) column for forecasting.", None)

    # Check for a datetime-parseable column
    if not _has_datetime_column(df):
        return (False,
                "No datetime-parseable column found in the CSV. The file must "
                "contain at least one column that can be parsed as dates/timestamps.", None)

    return True, "", df


def _has_datetime_column(df: pd.DataFrame) -> bool:
    """Check if the DataFrame contains at least one datetime-parseable column.

    Parameters:
        df: The pandas DataFrame to check.

    Returns:
        True if at least one datetime-parseable column exists, False otherwise.
    """
    # Check columns already typed as datetime
    if len(df.select_dtypes(include=["datetime64"]).columns) > 0:
        return True

    # Try to parse object/string columns as datetime
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            # Valid if at least 80% of non-null values parse successfully
            valid_ratio = parsed.notna().sum() / max(df[col].notna().sum(), 1)
            if valid_ratio >= 0.8:
                return True
        except (ValueError, TypeError):
            continue
    return False


def forecast(
    file, target_column: str, horizon: int,
) -> tuple[Optional[plt.Figure], str]:
    """Run probabilistic forecasting on the uploaded CSV data.

    Validates input, loads model, runs inference on last 512 steps of the
    selected column, and generates a plot with P10/P50/P90 intervals.

    Parameters:
        file: The uploaded CSV file object from Gradio.
        target_column: Name of the numeric column to forecast.
        horizon: Number of future time steps to predict (24-192).

    Returns:
        A tuple of (matplotlib Figure or None, status message string).
    """
    # Validate inputs
    if file is None:
        return None, "⚠️ Please upload a CSV file first."
    if not target_column:
        return None, "⚠️ Please select a target column for forecasting."

    # Get file path and validate CSV
    file_path = file.name if hasattr(file, "name") else file
    is_valid, error_msg, df = validate_csv(file_path)
    if not is_valid:
        return None, f"❌ {error_msg}"

    # Verify selected column exists and is numeric
    if target_column not in df.columns:
        return None, f"❌ Column '{target_column}' not found in the uploaded file."
    if not np.issubdtype(df[target_column].dtype, np.number):
        return None, f"❌ Column '{target_column}' is not numeric."

    # Prepare data: extract last 512 steps, handle NaN values
    series = df[target_column].values.astype(np.float64)
    series = pd.Series(series).ffill().bfill().values
    context_data = series[-Config.CONTEXT_LENGTH:]

    # Normalize context data using its own statistics
    norm_stats = compute_normalization_stats(context_data)
    normalized_context = normalize(context_data, norm_stats)

    # Run inference with timeout protection
    try:
        start_time = time.time()
        model, head = load_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use custom head if horizon differs from default (96)
        if horizon != Config.FORECAST_HORIZON:
            custom_head = ProbabilisticForecastHead(
                d_model=Config.D_MODEL, num_patches=Config.NUM_PATCHES,
                forecast_horizon=horizon, quantiles=Config.QUANTILES,
            ).to(device).eval()
        else:
            custom_head = head

        # Forward pass: no gradients needed for inference
        with torch.no_grad():
            context_tensor = torch.tensor(
                normalized_context, dtype=torch.float32, device=device
            ).unsqueeze(0)  # Shape: (1, 512)
            encoder_output = model(context_tensor)  # Shape: (1, 63, 256)
            quantile_forecasts = custom_head(encoder_output)  # Shape: (1, horizon, 3)
            forecast_np = quantile_forecasts.squeeze(0).cpu().numpy()

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > INFERENCE_TIMEOUT_SECONDS:
            return None, (f"❌ Inference timed out after {elapsed:.1f} seconds "
                          f"(limit: {INFERENCE_TIMEOUT_SECONDS}s). Please try again.")

        # Inverse normalize forecasts to original scale
        forecast_original = inverse_normalize(forecast_np, norm_stats)
        p10, p50, p90 = forecast_original[:, 0], forecast_original[:, 1], forecast_original[:, 2]

        # Create the forecast plot
        fig = _create_forecast_plot(context_data, p10, p50, p90, target_column, horizon)
        elapsed = time.time() - start_time
        return fig, (f"✅ Forecast complete! Generated {horizon}-step probabilistic "
                     f"forecast for '{target_column}' in {elapsed:.1f}s.")

    except Exception as e:
        return None, f"❌ Inference failed: {str(e)}. Please try again."


def _create_forecast_plot(
    context_data: np.ndarray, p10: np.ndarray, p50: np.ndarray,
    p90: np.ndarray, target_column: str, horizon: int,
) -> plt.Figure:
    """Create a matplotlib figure showing historical context and forecast.

    Displays historical context as blue line, P50 forecast as orange line,
    and shaded P10-P90 region (80% prediction interval).

    Parameters:
        context_data: Historical values (last 512 steps, original scale).
        p10: P10 (10th percentile) forecast values.
        p50: P50 (median) forecast values.
        p90: P90 (90th percentile) forecast values.
        target_column: Name of the target column (for plot title).
        horizon: Number of forecast steps.

    Returns:
        A matplotlib Figure object containing the forecast visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    context_len = len(context_data)
    context_x = np.arange(context_len)
    forecast_x = np.arange(context_len, context_len + horizon)

    # Plot historical context, P50 forecast, and P10-P90 interval
    ax.plot(context_x, context_data, color="steelblue", linewidth=1.5,
            label="Historical Context")
    ax.plot(forecast_x, p50, color="darkorange", linewidth=2.0,
            label="P50 Forecast (Median)")
    ax.fill_between(forecast_x, p10, p90, alpha=0.3, color="orange",
                    label="P10-P90 Interval (80%)")

    # Vertical line separating history from forecast
    ax.axvline(x=context_len, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    # Labels, title, legend, and grid
    ax.set_xlabel("Time Steps", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(f"Probabilistic Forecast — {target_column} "
                 f"(Context: {context_len} steps, Horizon: {horizon} steps)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def on_file_upload(file) -> dict:
    """Handle file upload: populate column dropdown with numeric columns.

    Parameters:
        file: The uploaded file object from Gradio.

    Returns:
        A Gradio update dictionary for the dropdown component.
    """
    if file is None:
        return gr.update(choices=[], value=None)
    try:
        file_path = file.name if hasattr(file, "name") else file
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return gr.update(choices=[], value=None)
        return gr.update(choices=numeric_cols, value=numeric_cols[0])
    except Exception:
        return gr.update(choices=[], value=None)


def create_app() -> gr.Blocks:
    """Create and configure the Gradio Blocks application.

    Builds the UI with file upload, column dropdown, horizon slider,
    forecast button, plot output, and status textbox.

    Returns:
        A configured gr.Blocks application ready to launch.
    """
    with gr.Blocks(title="Time Series Foundation Model — Probabilistic Forecasting") as app:
        gr.Markdown("""
            # 🔮 Time Series Foundation Model
            ## Probabilistic Forecasting Demo

            Upload a CSV file with time series data to generate probabilistic
            forecasts (P10/P50/P90 prediction intervals) using a pretrained PatchTST model.

            **Requirements:** CSV with ≥512 rows, a datetime column, and ≥1 numeric column.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # File upload widget accepting CSV files up to 50MB
                file_input = gr.File(
                    label="Upload CSV File (max 50MB)", file_types=[".csv"], type="filepath")
                # Dropdown for target numeric column selection
                column_dropdown = gr.Dropdown(
                    label="Target Column", choices=[], value=None,
                    interactive=True, info="Select the numeric column to forecast")
                # Slider for forecast horizon (24 to 192 steps)
                horizon_slider = gr.Slider(
                    minimum=24, maximum=192, value=96, step=1,
                    label="Forecast Horizon (steps)",
                    info="Number of future time steps to predict")
                # Forecast button triggers inference
                forecast_btn = gr.Button("🚀 Forecast", variant="primary", size="lg")

            with gr.Column(scale=2):
                # Plot output for forecast visualization
                plot_output = gr.Plot(label="Forecast Results")
                # Status message textbox for errors and success messages
                status_output = gr.Textbox(label="Status", interactive=False, lines=2)

        # Event: file upload populates column dropdown
        file_input.change(fn=on_file_upload, inputs=[file_input], outputs=[column_dropdown])
        # Event: forecast button runs inference
        forecast_btn.click(
            fn=forecast, inputs=[file_input, column_dropdown, horizon_slider],
            outputs=[plot_output, status_output])

    return app


# Entry point for HuggingFace Space deployment — run with `python app/app.py`
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
