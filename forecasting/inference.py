"""Zero-shot forecasting inference module for the Time Series Foundation Model.

This module implements the zero-shot inference pipeline: it loads a pretrained
PatchTST model, sets it to evaluation mode (no dropout, no gradient computation),
and generates probabilistic forecasts (P10/P50/P90) on unseen data using a sliding
window approach. Predictions are returned in the original data scale after inverse
normalization.

Related modules:
    - model/patchtst.py provides the PatchTSTModel encoder backbone.
    - forecasting/probabilistic_head.py provides the ProbabilisticForecastHead
      that maps encoder output to quantile predictions.
    - data/preprocess.py provides inverse_normalize to convert predictions back
      to the original data scale.
    - config.py supplies CONTEXT_LENGTH (512), FORECAST_HORIZON (96), and other
      hyperparameters used for sliding window inference.
"""

import numpy as np
import torch

from config import Config
from data.preprocess import inverse_normalize
from model.patchtst import PatchTSTModel
from forecasting.probabilistic_head import ProbabilisticForecastHead


def compute_num_windows(
    data_length: int,
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
) -> int:
    """Calculate the number of non-overlapping forecast windows for a given data length.

    The sliding window moves with a stride equal to the forecast horizon (96 steps),
    producing non-overlapping forecast regions. Each window requires context_length
    (512) input steps followed by forecast_horizon (96) target steps.

    Parameters:
        data_length: Total number of time steps in the data array.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to predict per window (default 96).
        stride: Step size between consecutive windows (default 96, non-overlapping).

    Returns:
        The number of valid evaluation windows: floor((T - context - horizon) / stride) + 1.
        Returns 0 if the data is too short for even one window.
    """
    # Minimum data length needed for at least one window
    min_required = context_length + forecast_horizon

    # If data is too short, no windows can be formed
    if data_length < min_required:
        return 0

    # Number of windows: floor((T - context - horizon) / stride) + 1
    num_windows = (data_length - context_length - forecast_horizon) // stride + 1

    return num_windows


def zero_shot_forecast(
    model: PatchTSTModel,
    head: ProbabilisticForecastHead,
    data: np.ndarray,
    norm_stats: dict[str, list[float]],
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
    device: str = "cpu",
) -> np.ndarray:
    """Generate probabilistic forecasts without any fine-tuning (zero-shot).

    This function implements the full zero-shot inference pipeline:
    1. Set model and head to evaluation mode (disables dropout)
    2. Use torch.no_grad() to disable gradient computation for efficiency
    3. Slide a window across the input data with the specified stride
    4. For each window, feed the context through the encoder and forecast head
    5. Apply inverse normalization to return predictions in the original scale

    The sliding window uses context=512, horizon=96, stride=96 by default,
    producing non-overlapping forecast regions that tile the test set.

    Parameters:
        model: A pretrained PatchTSTModel instance (encoder backbone).
        head: A ProbabilisticForecastHead instance for quantile prediction.
        data: A 1D numpy array of normalized time series values (the test split).
              Must have length >= context_length + forecast_horizon for at least
              one forecast window.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
                    As returned by compute_normalization_stats in data/preprocess.py.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to predict (default 96).
        stride: Step size between consecutive windows (default 96, non-overlapping).
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        A numpy array of shape (num_windows, forecast_horizon, 3) containing
        P10/P50/P90 quantile forecasts in the original data scale.
        The three channels along axis 2 correspond to [P10, P50, P90].

    Raises:
        ValueError: If the data is too short to form at least one forecast window.
    """
    # -------------------------------------------------------------------------
    # Validate that the data is long enough for at least one forecast window
    # -------------------------------------------------------------------------
    num_windows = compute_num_windows(
        data_length=len(data),
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if num_windows == 0:
        raise ValueError(
            f"Input data length ({len(data)}) is too short for zero-shot forecasting. "
            f"Need at least {context_length + forecast_horizon} time steps "
            f"(context_length={context_length} + forecast_horizon={forecast_horizon})."
        )

    # -------------------------------------------------------------------------
    # Set model and head to evaluation mode
    # This disables dropout layers and sets batch normalization to use running
    # statistics instead of batch statistics (if any). No parameter updates occur.
    # -------------------------------------------------------------------------
    model.eval()
    head.eval()

    # Move model and head to the specified device (CPU or GPU)
    model = model.to(device)
    head = head.to(device)

    # -------------------------------------------------------------------------
    # Collect forecasts for all windows using no-gradient context
    # torch.no_grad() disables gradient tracking, reducing memory usage and
    # speeding up inference since we never need to backpropagate.
    # -------------------------------------------------------------------------
    all_forecasts: list[np.ndarray] = []

    with torch.no_grad():
        for window_idx in range(num_windows):
            # -----------------------------------------------------------------
            # Extract the context window for this forecast
            # Start position advances by stride (96) for each window
            # -----------------------------------------------------------------
            start = window_idx * stride
            end = start + context_length

            # Slice the normalized input data for this window
            context = data[start:end]

            # -----------------------------------------------------------------
            # Convert numpy context to a PyTorch tensor
            # Add batch dimension: (context_length,) → (1, context_length)
            # -----------------------------------------------------------------
            context_tensor = torch.tensor(
                context, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # -----------------------------------------------------------------
            # Forward pass through the encoder
            # Input: (1, 512) → Output: (1, 63, 256)
            # The encoder produces contextualized patch embeddings
            # -----------------------------------------------------------------
            encoder_output = model(context_tensor)

            # -----------------------------------------------------------------
            # Forward pass through the probabilistic forecast head
            # Input: (1, 63, 256) → Output: (1, 96, 3)
            # The head maps patch embeddings to P10/P50/P90 quantile forecasts
            # -----------------------------------------------------------------
            quantile_forecasts = head(encoder_output)

            # -----------------------------------------------------------------
            # Move predictions back to CPU and convert to numpy
            # Shape: (1, 96, 3) → (96, 3)
            # -----------------------------------------------------------------
            forecast_np = quantile_forecasts.squeeze(0).cpu().numpy()

            # -----------------------------------------------------------------
            # Apply inverse normalization to return predictions to original scale
            # The forecast head outputs predictions in the normalized space,
            # so we reverse the z-score transformation using the stored stats.
            # inverse_normalize expects shape (time_steps, channels) or (time_steps,)
            # Our forecast is (96, 3) which is treated as 3 "channels"
            # -----------------------------------------------------------------
            forecast_original_scale = inverse_normalize(forecast_np, norm_stats)

            # Store this window's forecast
            all_forecasts.append(forecast_original_scale)

    # -------------------------------------------------------------------------
    # Stack all window forecasts into a single array
    # Shape: (num_windows, forecast_horizon, 3) — one row per window
    # -------------------------------------------------------------------------
    result = np.stack(all_forecasts, axis=0)

    return result


def load_and_forecast(
    checkpoint_path: str,
    data: np.ndarray,
    norm_stats: dict[str, list[float]],
    device: str = "cpu",
) -> np.ndarray:
    """Convenience function: load a pretrained checkpoint and run zero-shot inference.

    This function handles the full workflow of loading model weights from a
    checkpoint file and generating forecasts, making it easy to use from
    evaluation scripts or the Gradio app.

    Parameters:
        checkpoint_path: Path to the saved checkpoint file (.pt format).
                         Must contain 'model_state_dict' key.
        data: A 1D numpy array of normalized time series values.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        A numpy array of shape (num_windows, forecast_horizon, 3) containing
        P10/P50/P90 quantile forecasts in the original data scale.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the data is too short for at least one forecast window.
    """
    # -------------------------------------------------------------------------
    # Load the checkpoint from disk
    # The checkpoint contains model weights, optimizer state, and metadata
    # -------------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # -------------------------------------------------------------------------
    # Instantiate the model and forecast head with default configuration
    # -------------------------------------------------------------------------
    model = PatchTSTModel(Config)
    head = ProbabilisticForecastHead(
        d_model=Config.D_MODEL,
        num_patches=Config.NUM_PATCHES,
        forecast_horizon=Config.FORECAST_HORIZON,
        quantiles=Config.QUANTILES,
    )

    # -------------------------------------------------------------------------
    # Load pretrained weights into the model
    # The checkpoint may store model weights under 'model_state_dict' key
    # -------------------------------------------------------------------------
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If the checkpoint is just a raw state dict, load it directly
        model.load_state_dict(checkpoint)

    # -------------------------------------------------------------------------
    # Load forecast head weights if available in the checkpoint
    # During pretraining, the head may not be saved; in that case we use
    # randomly initialized head weights (acceptable for zero-shot evaluation
    # if the head was trained separately or is part of the checkpoint)
    # -------------------------------------------------------------------------
    if "head_state_dict" in checkpoint:
        head.load_state_dict(checkpoint["head_state_dict"])

    # -------------------------------------------------------------------------
    # Run zero-shot inference using the loaded model and head
    # -------------------------------------------------------------------------
    forecasts = zero_shot_forecast(
        model=model,
        head=head,
        data=data,
        norm_stats=norm_stats,
        device=device,
    )

    return forecasts
