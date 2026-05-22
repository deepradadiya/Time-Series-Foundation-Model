"""Probabilistic forecasting head for quantile regression (P10/P50/P90).

This module implements the ProbabilisticForecastHead that maps encoder output
embeddings to quantile predictions for the forecast horizon. It produces three
quantile forecasts (P10, P50, P90) with monotonicity enforcement, providing
calibrated uncertainty estimates for time series forecasting.

Related modules:
    - model/patchtst.py provides the encoder output consumed by this head.
    - forecasting/inference.py uses this head for zero-shot forecasting.
    - forecasting/finetune.py trains this head on ETTh1 with the encoder frozen.
    - config.py supplies FORECAST_HORIZON (96), QUANTILES, D_MODEL, NUM_PATCHES.
"""

import torch
import torch.nn as nn

from config import Config


class ProbabilisticForecastHead(nn.Module):
    """Maps encoder patch embeddings to probabilistic quantile forecasts.

    This head takes the contextualized patch embeddings from the PatchTST encoder
    and produces three quantile predictions (P10, P50, P90) for each time step in
    the forecast horizon. Monotonicity is enforced by sorting quantile values at
    each time step so that P10 <= P50 <= P90.

    Architecture:
        Input: (batch, num_patches, d_model) = (B, 63, 256)
        → Flatten: (B, 63 * 256) = (B, 16128)
        → Linear: (B, forecast_horizon * num_quantiles) = (B, 96 * 3) = (B, 288)
        → Reshape: (B, 96, 3)
        → Sort along quantile dim for monotonicity
        Output: (B, 96, 3) with P10 <= P50 <= P90

    Args:
        d_model: Dimension of encoder output embeddings (default 256).
        num_patches: Number of patches from the encoder (default 63).
        forecast_horizon: Number of future time steps to predict (default 96).
        quantiles: List of quantile levels to predict (default [0.1, 0.5, 0.9]).

    Example:
        >>> head = ProbabilisticForecastHead(d_model=256, num_patches=63,
        ...                                  forecast_horizon=96, quantiles=[0.1, 0.5, 0.9])
        >>> encoder_out = torch.randn(4, 63, 256)
        >>> forecasts = head(encoder_out)  # shape: (4, 96, 3)
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        num_patches: int = Config.NUM_PATCHES,
        forecast_horizon: int = Config.FORECAST_HORIZON,
        quantiles: list[float] = Config.QUANTILES,
    ) -> None:
        """Initialize the probabilistic forecast head.

        Args:
            d_model: Embedding dimension from the encoder (default 256).
            num_patches: Number of patch embeddings expected (default 63).
            forecast_horizon: Number of future steps to predict (default 96).
            quantiles: Quantile levels for prediction (default [0.1, 0.5, 0.9]).
        """
        super().__init__()

        # Store configuration for validation and reshaping
        self.d_model = d_model
        self.num_patches = num_patches
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # -----------------------------------------------------------------------
        # Compute input and output dimensions for the linear projection
        # Input: flattened encoder output (num_patches * d_model)
        # Output: forecast_horizon * num_quantiles (96 * 3 = 288)
        # -----------------------------------------------------------------------
        input_dim = num_patches * d_model
        output_dim = forecast_horizon * self.num_quantiles

        # -----------------------------------------------------------------------
        # Linear projection from flattened patch embeddings to quantile forecasts
        # This single linear layer maps the full context representation to all
        # forecast steps and quantiles simultaneously.
        # -----------------------------------------------------------------------
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Map encoder output to quantile forecasts with monotonicity enforcement.

        Args:
            encoder_output: Tensor of shape (batch, num_patches, d_model).
                            For standard config: (B, 63, 256).

        Returns:
            Tensor of shape (batch, forecast_horizon, num_quantiles) = (B, 96, 3).
            Values are sorted along the quantile dimension so P10 <= P50 <= P90.

        Raises:
            ValueError: If the input sequence length does not match num_patches (63),
                        indicating the input does not correspond to a valid context
                        window of 512 time steps.
        """
        # -----------------------------------------------------------------------
        # Validate input dimensions
        # The encoder output must have exactly num_patches (63) patch embeddings,
        # which corresponds to a context window of 512 time steps.
        # -----------------------------------------------------------------------
        if encoder_output.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected encoder output with {self.num_patches} patches "
                f"(from context window of 512 time steps), but got "
                f"{encoder_output.shape[1]} patches. Input dimensions: "
                f"{encoder_output.shape}."
            )

        # -----------------------------------------------------------------------
        # Flatten the patch embeddings into a single vector per sample
        # Shape: (batch, num_patches, d_model) → (batch, num_patches * d_model)
        # This concatenates all patch representations into one long vector
        # -----------------------------------------------------------------------
        batch_size = encoder_output.shape[0]
        flat = encoder_output.reshape(batch_size, -1)

        # -----------------------------------------------------------------------
        # Project to forecast space
        # Shape: (batch, num_patches * d_model) → (batch, forecast_horizon * num_quantiles)
        # -----------------------------------------------------------------------
        raw_output = self.projection(flat)

        # -----------------------------------------------------------------------
        # Reshape to separate forecast steps and quantiles
        # Shape: (batch, forecast_horizon * num_quantiles) → (batch, forecast_horizon, num_quantiles)
        # -----------------------------------------------------------------------
        reshaped = raw_output.reshape(batch_size, self.forecast_horizon, self.num_quantiles)

        # -----------------------------------------------------------------------
        # Enforce monotonicity: sort quantile values at each time step
        # This ensures P10 <= P50 <= P90 by sorting along the last dimension.
        # Any violations from the raw linear output are corrected here.
        # -----------------------------------------------------------------------
        sorted_output, _ = torch.sort(reshaped, dim=-1)

        return sorted_output


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list[float] = Config.QUANTILES,
) -> torch.Tensor:
    """Compute pinball loss (quantile regression loss) for probabilistic forecasts.

    The pinball loss penalizes under-predictions and over-predictions asymmetrically
    based on the quantile level. For quantile tau:
        loss = tau * max(y - q_hat, 0) + (1 - tau) * max(q_hat - y, 0)

    The total loss is averaged across all quantiles, time steps, and batch samples.

    Parameters:
        predictions: Tensor of shape (batch, forecast_horizon, num_quantiles).
                     Contains the predicted quantile values (e.g., P10, P50, P90).
        targets: Tensor of shape (batch, forecast_horizon) or (batch, forecast_horizon, 1).
                 Contains the actual observed values.
        quantiles: List of quantile levels (default [0.1, 0.5, 0.9]).

    Returns:
        A scalar tensor containing the mean pinball loss across all quantiles,
        time steps, and batch samples.
    """
    # -----------------------------------------------------------------------
    # Ensure targets have the right shape for broadcasting
    # If targets is (batch, horizon), expand to (batch, horizon, 1) for
    # element-wise comparison with each quantile prediction
    # -----------------------------------------------------------------------
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    # -----------------------------------------------------------------------
    # Compute the error (residual) between actual values and predictions
    # Shape: (batch, forecast_horizon, num_quantiles)
    # -----------------------------------------------------------------------
    errors = targets - predictions

    # -----------------------------------------------------------------------
    # Compute pinball loss for each quantile level
    # For each quantile tau:
    #   - If y > q_hat (under-prediction): loss = tau * (y - q_hat)
    #   - If y <= q_hat (over-prediction): loss = (1 - tau) * (q_hat - y)
    # -----------------------------------------------------------------------
    quantile_tensor = torch.tensor(quantiles, dtype=predictions.dtype, device=predictions.device)

    # Positive errors (under-prediction): weighted by tau
    positive_loss = quantile_tensor * torch.clamp(errors, min=0.0)

    # Negative errors (over-prediction): weighted by (1 - tau)
    negative_loss = (1.0 - quantile_tensor) * torch.clamp(-errors, min=0.0)

    # -----------------------------------------------------------------------
    # Total pinball loss: sum of positive and negative components
    # Average across all dimensions (batch, time steps, quantiles)
    # -----------------------------------------------------------------------
    total_loss = (positive_loss + negative_loss).mean()

    return total_loss
