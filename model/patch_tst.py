"""PatchTST Model Assembly — Full end-to-end time series forecasting model.

This module composes all PatchTST components into a single nn.Module that takes
raw univariate time series and produces probabilistic quantile forecasts
(P10/P50/P90). It provides two interfaces:

    - forward(): For training — returns normalized quantile predictions
    - forecast(): For inference — returns denormalized P10/P50/P90 predictions

================================================================================
FULL PIPELINE ARCHITECTURE
================================================================================

Input: (batch, seq_len=512) — raw univariate time series (z-score normalized)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PatchEmbedding (model/patching.py)                                         │
│  - Unfold into overlapping patches: (B, 512) → (B, 63, 16)                 │
│  - Linear projection to embedding space: (B, 63, 16) → (B, 63, 256)        │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SinusoidalPositionalEncoding (model/positional_encoding.py)                 │
│  - Add fixed sin/cos position signals: (B, 63, 256) → (B, 63, 256)         │
│  - Dropout for regularization                                               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TransformerEncoder (model/transformer_encoder.py)                           │
│  - 6× TransformerEncoderBlock (MHSA + FFN + residuals)                      │
│  - Final LayerNorm                                                          │
│  - (B, 63, 256) → (B, 63, 256)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ProbabilisticHead (model/probabilistic_head.py)                             │
│  - 3 separate linear heads (P10, P50, P90)                                  │
│  - Monotonicity enforcement via sort                                        │
│  - (B, 63, 256) → (B, 96, 3)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output: (batch, forecast_horizon=96, 3) — quantile forecasts [P10, P50, P90]

================================================================================
USAGE
================================================================================

Training:
    model = PatchTST()
    output = model(x)  # x: (B, 512), output: (B, 96, 3)
    loss = quantile_loss(output, targets)

Inference:
    model = PatchTST()
    model.eval()
    result = model.forecast(x, mean=series_mean, std=series_std)
    # result['p10']: (B, 96) — denormalized lower bound
    # result['p50']: (B, 96) — denormalized median forecast
    # result['p90']: (B, 96) — denormalized upper bound

================================================================================

Related modules:
    - model/patching.py: PatchEmbedding (stage 1)
    - model/positional_encoding.py: SinusoidalPositionalEncoding (stage 2)
    - model/transformer_encoder.py: TransformerEncoder (stage 3)
    - model/probabilistic_head.py: ProbabilisticHead (stage 4)
    - config.py: All hyperparameters
"""

import torch
import torch.nn as nn

from config import Config
from model.patching import PatchEmbedding
from model.positional_encoding import SinusoidalPositionalEncoding
from model.transformer_encoder import TransformerEncoder
from model.probabilistic_head import ProbabilisticHead


class PatchTST(nn.Module):
    """Complete PatchTST model for probabilistic time series forecasting.

    Composes PatchEmbedding, SinusoidalPositionalEncoding, TransformerEncoder,
    and ProbabilisticHead into a single end-to-end model. Provides both a
    training interface (forward) and an inference interface (forecast).

    At initialization, the model counts and prints its total trainable
    parameters, with a warning if the count falls outside the target
    range of 8-12 million parameters (suitable for Colab T4 GPU training).

    Args:
        config: Configuration class with model hyperparameters.
            Expected attributes: D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
            PATCH_LEN, PATCH_STRIDE, NUM_PATCHES, FORECAST_HORIZON, QUANTILES.

    Example:
        >>> model = PatchTST()
        >>> x = torch.randn(4, 512)  # batch of 4 time series
        >>> output = model(x)         # shape: (4, 96, 3)
        >>> result = model.forecast(x, mean=0.5, std=1.2)
        >>> result['p50'].shape       # (4, 96)
    """

    def __init__(self, config=Config) -> None:
        """Initialize PatchTST with all submodules from config.

        Instantiates:
            - PatchEmbedding: patches raw series and projects to d_model
            - SinusoidalPositionalEncoding: adds position information
            - TransformerEncoder: 6-layer encoder stack
            - ProbabilisticHead: maps to P10/P50/P90 quantile forecasts

        After instantiation, prints the total parameter count and warns
        if it falls outside the 8-12M target range.

        Args:
            config: Configuration class with all hyperparameters.
        """
        super().__init__()

        # Store config for reference
        self.config = config

        # -----------------------------------------------------------------------
        # Stage 1: Patch Embedding
        # Segments raw time series into overlapping patches and projects each
        # patch to the transformer's embedding dimension.
        # Input:  (batch, seq_len)
        # Output: (batch, num_patches, d_model)
        # -----------------------------------------------------------------------
        self.patch_embedding = PatchEmbedding(
            patch_len=config.PATCH_LEN,
            d_model=config.D_MODEL,
            stride=config.PATCH_STRIDE,
        )

        # -----------------------------------------------------------------------
        # Stage 2: Sinusoidal Positional Encoding
        # Adds fixed sin/cos position signals so the transformer can distinguish
        # patch order. Registered as a buffer (no gradients).
        # Input:  (batch, num_patches, d_model)
        # Output: (batch, num_patches, d_model)
        # -----------------------------------------------------------------------
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.D_MODEL,
            max_len=128,
            dropout=config.DROPOUT,
        )

        # -----------------------------------------------------------------------
        # Stage 3: Transformer Encoder
        # Stack of 6 encoder blocks with multi-head self-attention, FFN,
        # LayerNorm, and residual connections. Learns cross-patch relationships.
        # Input:  (batch, num_patches, d_model)
        # Output: (batch, num_patches, d_model)
        # -----------------------------------------------------------------------
        self.encoder = TransformerEncoder(
            n_layers=config.N_LAYERS,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
        )

        # -----------------------------------------------------------------------
        # Stage 4: Probabilistic Head
        # Three separate linear heads producing P10/P50/P90 quantile forecasts.
        # Monotonicity enforced via sorting.
        # Input:  (batch, num_patches, d_model)
        # Output: (batch, forecast_horizon, 3)
        # -----------------------------------------------------------------------
        self.head = ProbabilisticHead(
            d_model=config.D_MODEL,
            num_patches=config.NUM_PATCHES,
            forecast_horizon=config.FORECAST_HORIZON,
            quantiles=config.QUANTILES,
        )

        # -----------------------------------------------------------------------
        # Parameter Count Report
        # Print total trainable parameters and warn if outside 8-12M target.
        # This budget is chosen for practical training on a Colab T4 GPU.
        # -----------------------------------------------------------------------
        num_params = self.count_parameters()
        print(f"PatchTST initialized with {num_params:,} trainable parameters")

        if num_params < 8_000_000 or num_params > 12_000_000:
            print(
                f"WARNING: Parameter count ({num_params:,}) is outside the "
                f"target range of 8-12 million. Consider adjusting model "
                f"dimensions (d_model, n_layers, d_ff) to fit within budget."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw time series → quantile forecasts (normalized space).

        Pipeline:
            input (batch, seq_len)
            → patch_embedding → (batch, num_patches, d_model)
            → positional_encoding → (batch, num_patches, d_model)
            → encoder → (batch, num_patches, d_model)
            → head → (batch, forecast_horizon, 3)

        Args:
            x: Input tensor of shape (batch, seq_len) containing raw
                (z-score normalized) univariate time series.

        Returns:
            Tensor of shape (batch, forecast_horizon, 3) containing
            quantile predictions [P10, P50, P90] in normalized space.
        """
        # Stage 1: Patch and embed
        # (batch, seq_len) → (batch, num_patches, d_model)
        x = self.patch_embedding(x)

        # Stage 2: Add positional encoding
        # (batch, num_patches, d_model) → (batch, num_patches, d_model)
        x = self.positional_encoding(x)

        # Stage 3: Transformer encoder
        # (batch, num_patches, d_model) → (batch, num_patches, d_model)
        x = self.encoder(x)

        # Stage 4: Probabilistic head
        # (batch, num_patches, d_model) → (batch, forecast_horizon, 3)
        x = self.head(x)

        return x

    def forecast(
        self,
        x: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> dict:
        """Inference method: raw series → denormalized P10/P50/P90 predictions.

        Calls forward() to get normalized quantile forecasts, then splits
        the output into separate P10/P50/P90 channels and denormalizes
        each using the provided mean and std (reversing z-score normalization).

        Denormalization formula:
            denormalized = normalized * std + mean

        Args:
            x: Input tensor of shape (batch, seq_len) containing raw
                (z-score normalized) univariate time series.
            mean: Mean of the original (un-normalized) time series.
                Used to shift predictions back to original scale.
            std: Standard deviation of the original time series.
                Used to scale predictions back to original magnitude.

        Returns:
            Dictionary with keys 'p10', 'p50', 'p90', each containing a
            tensor of shape (batch, forecast_horizon) with denormalized
            quantile predictions.
        """
        # Get normalized quantile forecasts: (batch, forecast_horizon, 3)
        output = self.forward(x)

        # Split into individual quantile channels
        # Each has shape (batch, forecast_horizon)
        p10 = output[:, :, 0]  # 10th percentile (lower bound)
        p50 = output[:, :, 1]  # 50th percentile (median)
        p90 = output[:, :, 2]  # 90th percentile (upper bound)

        # Denormalize: reverse z-score normalization
        # normalized = (original - mean) / std
        # original = normalized * std + mean
        p10 = p10 * std + mean
        p50 = p50 * std + mean
        p90 = p90 * std + mean

        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters in the model.

        Iterates over all parameters that require gradient computation
        and sums their element counts (numel).

        Returns:
            Integer count of all trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
