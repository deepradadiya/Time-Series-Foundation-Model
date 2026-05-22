"""Probabilistic forecasting head with separate quantile heads (P10/P50/P90).

This module implements the ProbabilisticHead that maps transformer encoder output
to three quantile predictions for the forecast horizon. Unlike a single linear
projection, this design uses three independent nn.Linear heads — one per quantile
— allowing each head to specialize in its respective region of the predictive
distribution.

Probabilistic Forecasting — What the Quantiles Mean:
─────────────────────────────────────────────────────
  P10 (tau=0.1): Lower bound of the prediction interval.
      There is a 10% chance the actual value falls BELOW this prediction.
      Think of it as the optimistic lower fence — actual values rarely go below.

  P50 (tau=0.5): Median prediction (central estimate).
      There is a 50% chance the actual value is above or below this prediction.
      This is the model's "best guess" — symmetric penalty for over/under.

  P90 (tau=0.9): Upper bound of the prediction interval.
      There is a 90% chance the actual value falls BELOW this prediction.
      Think of it as the conservative upper fence — actual values rarely exceed.

  Together, P10 and P90 form an 80% prediction interval:
      ┌─────────────────────────────────────────────────────┐
      │         P10          P50          P90               │
      │          │            │            │                │
      │    ──────┼────────────┼────────────┼──────          │
      │          │◄──── 80% interval ─────►│                │
      │          │            │            │                │
      │  10% of  │   80% of actual values  │  10% of       │
      │  actuals │   fall in this range    │  actuals      │
      │  below   │                         │  above        │
      └─────────────────────────────────────────────────────┘

  Monotonicity Constraint: P10 <= P50 <= P90 must always hold.
  We enforce this via torch.sort along the quantile dimension after
  the raw linear projections, guaranteeing valid prediction intervals.

Architecture:
    Input: (batch, num_patches, d_model) = (B, 63, 256)
    → Flatten: (B, 63 * 256) = (B, 16128)
    → head_p10: (B, 16128) → (B, 96)   [lower bound]
    → head_p50: (B, 16128) → (B, 96)   [median]
    → head_p90: (B, 16128) → (B, 96)   [upper bound]
    → Stack: (B, 96, 3)
    → Sort dim=-1 for monotonicity
    Output: (B, 96, 3) with P10 <= P50 <= P90

Related modules:
    - model/transformer_encoder.py provides the encoder output consumed here.
    - model/patch_tst.py assembles this head into the full PatchTST model.
    - config.py supplies FORECAST_HORIZON (96), QUANTILES, D_MODEL, NUM_PATCHES.
"""

import torch
import torch.nn as nn

from config import Config


class ProbabilisticHead(nn.Module):
    """Maps encoder patch embeddings to probabilistic quantile forecasts.

    Uses three separate linear heads — one per quantile — so each head can
    independently learn to predict its respective quantile level. This is
    preferable to a single shared projection because quantile boundaries
    (P10 vs P90) require fundamentally different learned mappings.

    Args:
        d_model: Dimension of encoder output embeddings (default 256).
        num_patches: Number of patches from the encoder (default 63).
        forecast_horizon: Number of future time steps to predict (default 96).
        quantiles: List of quantile levels to predict (default [0.1, 0.5, 0.9]).

    Example:
        >>> head = ProbabilisticHead(d_model=256, num_patches=63, forecast_horizon=96)
        >>> encoder_out = torch.randn(4, 63, 256)
        >>> forecasts = head(encoder_out)  # shape: (4, 96, 3)
        >>> assert (forecasts[:, :, 0] <= forecasts[:, :, 1]).all()  # P10 <= P50
        >>> assert (forecasts[:, :, 1] <= forecasts[:, :, 2]).all()  # P50 <= P90
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        num_patches: int = Config.NUM_PATCHES,
        forecast_horizon: int = Config.FORECAST_HORIZON,
        quantiles: list = Config.QUANTILES,
    ) -> None:
        """Initialize the probabilistic head with three separate linear projections.

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

        # -----------------------------------------------------------------------
        # Input dimension: flattened encoder output (num_patches * d_model)
        # For default config: 63 * 256 = 16128
        # -----------------------------------------------------------------------
        input_dim = num_patches * d_model

        # -----------------------------------------------------------------------
        # Three separate linear heads — one per quantile
        #
        # Why separate heads instead of one shared projection?
        # Each quantile captures a different region of the predictive distribution:
        #   - head_p10 learns to predict the 10th percentile (lower bound)
        #   - head_p50 learns to predict the median (central tendency)
        #   - head_p90 learns to predict the 90th percentile (upper bound)
        #
        # Separate heads allow each to develop specialized weight patterns.
        # The P10 head might focus on features indicating downside risk,
        # while P90 focuses on features indicating upside potential.
        # -----------------------------------------------------------------------
        self.head_p10 = nn.Linear(input_dim, forecast_horizon)
        self.head_p50 = nn.Linear(input_dim, forecast_horizon)
        self.head_p90 = nn.Linear(input_dim, forecast_horizon)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Map encoder output to quantile forecasts with monotonicity enforcement.

        Args:
            encoder_output: Tensor of shape (batch, num_patches, d_model).
                            For standard config: (B, 63, 256).

        Returns:
            Tensor of shape (batch, forecast_horizon, 3) = (B, 96, 3).
            The last dimension contains [P10, P50, P90] sorted to ensure
            monotonicity: P10 <= P50 <= P90 at every forecast timestep.

        Raises:
            ValueError: If encoder_output.shape[1] != num_patches, indicating
                        the input does not match the expected number of patches.
        """
        # -----------------------------------------------------------------------
        # Input validation: ensure the encoder output has the expected number
        # of patches. A mismatch means the upstream context window or patching
        # configuration is inconsistent with this head's linear layer dimensions.
        # -----------------------------------------------------------------------
        if encoder_output.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected encoder output with {self.num_patches} patches, "
                f"but got {encoder_output.shape[1]}. "
                f"Input shape: {encoder_output.shape}. "
                f"Ensure the context window produces exactly {self.num_patches} patches."
            )

        # -----------------------------------------------------------------------
        # Flatten the patch embeddings into a single vector per sample
        # Shape: (batch, num_patches, d_model) → (batch, num_patches * d_model)
        # This concatenates all 63 patch representations (each 256-dim) into
        # one 16128-dim vector that captures the full context.
        # -----------------------------------------------------------------------
        batch_size = encoder_output.shape[0]
        flat = encoder_output.reshape(batch_size, -1)

        # -----------------------------------------------------------------------
        # Pass through each quantile head independently
        # Each head: (batch, num_patches * d_model) → (batch, forecast_horizon)
        # -----------------------------------------------------------------------
        p10 = self.head_p10(flat)  # (batch, forecast_horizon)
        p50 = self.head_p50(flat)  # (batch, forecast_horizon)
        p90 = self.head_p90(flat)  # (batch, forecast_horizon)

        # -----------------------------------------------------------------------
        # Stack quantile predictions along a new dimension
        # Shape: 3 × (batch, forecast_horizon) → (batch, forecast_horizon, 3)
        # The last dimension order is [P10, P50, P90]
        # -----------------------------------------------------------------------
        stacked = torch.stack([p10, p50, p90], dim=-1)

        # -----------------------------------------------------------------------
        # Enforce monotonicity: sort quantile values at each time step
        # This guarantees P10 <= P50 <= P90 by sorting along dim=-1.
        # Raw linear outputs may occasionally violate ordering (e.g., P10 > P50),
        # so sorting corrects any such violations post-hoc.
        # -----------------------------------------------------------------------
        sorted_output, _ = torch.sort(stacked, dim=-1)

        return sorted_output


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list = Config.QUANTILES,
) -> torch.Tensor:
    """Compute pinball loss (quantile regression loss) for probabilistic forecasts.

    The pinball loss asymmetrically penalizes under-predictions and over-predictions
    based on the quantile level tau. For each quantile:

        loss = tau * max(y - q_hat, 0) + (1 - tau) * max(q_hat - y, 0)

    Intuition:
    ──────────
    - For P10 (tau=0.1): Under-prediction penalty is low (0.1 weight),
      over-prediction penalty is high (0.9 weight). This pushes P10 downward,
      making it a conservative lower bound.

    - For P50 (tau=0.5): Symmetric penalty. Under and over-prediction are
      equally costly, producing the median estimate.

    - For P90 (tau=0.9): Under-prediction penalty is high (0.9 weight),
      over-prediction penalty is low (0.1 weight). This pushes P90 upward,
      making it a conservative upper bound.

    Args:
        predictions: Tensor of shape (batch, forecast_horizon, num_quantiles).
                     Contains the predicted quantile values (e.g., P10, P50, P90).
        targets: Tensor of shape (batch, forecast_horizon) or
                 (batch, forecast_horizon, 1). Contains actual observed values.
        quantiles: List of quantile levels (default [0.1, 0.5, 0.9]).

    Returns:
        A scalar tensor containing the mean pinball loss across all quantiles,
        time steps, and batch samples. Always non-negative.
    """
    # -----------------------------------------------------------------------
    # Ensure targets have the right shape for broadcasting
    # If targets is (batch, horizon), expand to (batch, horizon, 1) so we can
    # compute element-wise differences against each quantile prediction.
    # -----------------------------------------------------------------------
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    # -----------------------------------------------------------------------
    # Compute the error (residual) between actual values and predictions
    # errors > 0 means under-prediction (actual > predicted)
    # errors < 0 means over-prediction (actual < predicted)
    # Shape: (batch, forecast_horizon, num_quantiles)
    # -----------------------------------------------------------------------
    errors = targets - predictions

    # -----------------------------------------------------------------------
    # Create quantile tensor for vectorized computation
    # Shape: (num_quantiles,) — broadcasts against (batch, horizon, num_quantiles)
    # -----------------------------------------------------------------------
    quantile_tensor = torch.tensor(
        quantiles, dtype=predictions.dtype, device=predictions.device
    )

    # -----------------------------------------------------------------------
    # Pinball loss formula:
    #   loss = tau * max(y - q_hat, 0) + (1 - tau) * max(q_hat - y, 0)
    #
    # Decomposed:
    #   positive_loss = tau * max(errors, 0)       [under-prediction penalty]
    #   negative_loss = (1-tau) * max(-errors, 0)  [over-prediction penalty]
    # -----------------------------------------------------------------------
    positive_loss = quantile_tensor * torch.clamp(errors, min=0.0)
    negative_loss = (1.0 - quantile_tensor) * torch.clamp(-errors, min=0.0)

    # -----------------------------------------------------------------------
    # Total pinball loss: mean across all dimensions (batch, time, quantiles)
    # Returns a single scalar value for backpropagation.
    # -----------------------------------------------------------------------
    total_loss = (positive_loss + negative_loss).mean()

    return total_loss
