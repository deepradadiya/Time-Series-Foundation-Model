"""Top-level PatchTST model assembly: patch embedding + transformer encoder.

This module combines the patch embedding layer and the transformer encoder into
the complete PatchTST model. It implements the channel-independent design where
each univariate time series is processed independently through the same weights.
The model takes raw time series input of shape (batch, context_length) and produces
contextualized patch embeddings of shape (batch, num_patches, d_model).

Related modules:
    - model/patch_embedding.py provides the PatchEmbedding layer (projection + positional)
    - model/encoder.py provides the PatchTSTEncoder (6 transformer layers + final norm)
    - config.py supplies all architecture hyperparameters
    - pretraining/train.py uses this model for masked patch modeling pretraining
    - forecasting/inference.py uses this model for zero-shot forecasting
"""

import torch
import torch.nn as nn

from config import Config
from model.patch_embedding import PatchEmbedding
from model.encoder import PatchTSTEncoder


class PatchTSTModel(nn.Module):
    """Full PatchTST model: patch embedding followed by transformer encoder.

    This is the top-level model that takes a raw univariate time series as input,
    segments it into overlapping patches, projects them into embedding space with
    positional encoding, and processes them through a stack of transformer encoder
    layers. The output is a sequence of contextualized patch embeddings that can
    be consumed by task-specific heads (reconstruction head for pretraining,
    probabilistic head for forecasting).

    The model uses a channel-independent design: each univariate channel is
    processed independently through the same set of weights. This enables the
    model to generalize across domains with different numbers of channels.

    Architecture:
        Input (batch, 512) → Unfold into patches (batch, 63, 16)
                           → PatchEmbedding (batch, 63, 256)
                           → PatchTSTEncoder (batch, 63, 256)
                           → Output (batch, 63, 256)

    Args:
        config: Configuration object containing all hyperparameters.
            Uses: PATCH_LEN, PATCH_STRIDE, CONTEXT_LENGTH, NUM_PATCHES,
                  D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT.

    Example:
        >>> from config import Config
        >>> model = PatchTSTModel(Config)
        >>> x = torch.randn(4, 512)  # batch of 4 univariate series, 512 time steps
        >>> out = model(x)           # shape: (4, 63, 256)
        >>> print(model.count_parameters())  # < 10,000,000
    """

    def __init__(self, config: type = Config) -> None:
        """Initialize the PatchTST model with patch embedding and encoder.

        Args:
            config: Configuration class containing architecture hyperparameters.
                Must have attributes: PATCH_LEN, PATCH_STRIDE, CONTEXT_LENGTH,
                NUM_PATCHES, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT.
        """
        super().__init__()

        # -----------------------------------------------------------------------
        # Store configuration parameters for use in forward pass and validation
        # -----------------------------------------------------------------------
        self.patch_len = config.PATCH_LEN        # Length of each patch (16)
        self.patch_stride = config.PATCH_STRIDE  # Stride between patches (8)
        self.context_length = config.CONTEXT_LENGTH  # Expected input length (512)
        self.num_patches = config.NUM_PATCHES    # Number of patches produced (63)
        self.d_model = config.D_MODEL            # Embedding dimension (256)

        # -----------------------------------------------------------------------
        # Minimum input length validation threshold
        # The input must be at least patch_len (16) time steps long to produce
        # at least one complete patch. Inputs shorter than this are invalid.
        # -----------------------------------------------------------------------
        self.min_input_length = config.PATCH_LEN

        # -----------------------------------------------------------------------
        # Patch Embedding Layer
        # Converts raw patch vectors (length 16) into dense embeddings (dim 256)
        # and adds learnable positional encodings so the transformer can
        # distinguish patch positions within the sequence.
        # -----------------------------------------------------------------------
        self.patch_embedding = PatchEmbedding(
            patch_len=config.PATCH_LEN,
            d_model=config.D_MODEL,
            num_patches=config.NUM_PATCHES,
        )

        # -----------------------------------------------------------------------
        # Transformer Encoder
        # Stack of 6 pre-norm transformer layers with final LayerNorm.
        # Each layer applies multi-head self-attention (8 heads) and a feedforward
        # network (256 → 1024 → 256 with GELU) with residual connections.
        # This is where the model learns contextual relationships between patches.
        # -----------------------------------------------------------------------
        self.encoder = PatchTSTEncoder(
            n_layers=config.N_LAYERS,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a raw univariate time series through the full PatchTST model.

        The forward pass performs three steps:
        1. Validate input length (must be >= patch_len for at least one patch)
        2. Unfold the time series into overlapping patches using unfold operation
        3. Project patches to embeddings and process through transformer encoder

        Args:
            x: Input tensor of shape (batch_size, seq_length).
               For standard operation, seq_length = 512 (context_length).
               The model accepts any seq_length >= patch_len (16), but the
               standard configuration expects 512 time steps.

        Returns:
            Output tensor of shape (batch_size, num_patches, d_model).
            For standard input of length 512: (batch_size, 63, 256).
            Each output vector is a contextualized representation of one patch.

        Raises:
            ValueError: If the input sequence length is shorter than the minimum
                required length (patch_len = 16 time steps). The error message
                indicates the minimum required input length.
        """
        # -----------------------------------------------------------------------
        # Step 1: Validate input length
        # The input must have at least patch_len (16) time steps to form one
        # complete patch. Shorter inputs cannot be processed and raise an error.
        # -----------------------------------------------------------------------
        seq_length = x.shape[-1]

        if seq_length < self.min_input_length:
            raise ValueError(
                f"Input sequence length ({seq_length}) is shorter than the minimum "
                f"required length ({self.min_input_length}). The model needs at least "
                f"{self.min_input_length} time steps to create one complete patch."
            )

        # -----------------------------------------------------------------------
        # Step 2: Unfold the time series into overlapping patches
        # torch.Tensor.unfold(dimension, size, step) extracts sliding windows:
        #   - dimension=1: slide along the time axis
        #   - size=patch_len (16): each window is 16 time steps
        #   - step=patch_stride (8): windows overlap by 8 time steps
        #
        # Input shape:  (batch_size, seq_length)        e.g., (B, 512)
        # Output shape: (batch_size, num_patches, patch_len) e.g., (B, 63, 16)
        #
        # The number of patches is: floor((seq_length - patch_len) / stride) + 1
        # For seq_length=512: floor((512 - 16) / 8) + 1 = 63 patches
        # -----------------------------------------------------------------------
        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.patch_stride,
        )

        # -----------------------------------------------------------------------
        # Step 3: Patch Embedding
        # Project each raw patch (length 16) to a dense vector (dim 256) and
        # add learnable positional encodings. This transforms the patches from
        # raw time series segments into rich embeddings the transformer can process.
        #
        # Input shape:  (batch_size, num_patches, patch_len)  e.g., (B, 63, 16)
        # Output shape: (batch_size, num_patches, d_model)    e.g., (B, 63, 256)
        # -----------------------------------------------------------------------
        embedded = self.patch_embedding(patches)

        # -----------------------------------------------------------------------
        # Step 4: Transformer Encoder
        # Process the embedded patches through 6 transformer layers. Each layer
        # applies self-attention (allowing patches to attend to each other) and
        # a feedforward network (adding nonlinear transformations). The output
        # contains contextualized representations where each patch embedding
        # incorporates information from all other patches in the sequence.
        #
        # Input shape:  (batch_size, num_patches, d_model)  e.g., (B, 63, 256)
        # Output shape: (batch_size, num_patches, d_model)  e.g., (B, 63, 256)
        # -----------------------------------------------------------------------
        encoded = self.encoder(embedded)

        return encoded

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters in the model.

        This method sums up all parameters that require gradient computation
        (i.e., learnable parameters). It is used to verify the model stays
        within the 10M parameter budget required for Colab T4 GPU constraints.

        Returns:
            Total number of trainable parameters as an integer.
            For the default configuration, this should be well under 10,000,000.
        """
        # Sum the number of elements in each parameter tensor that requires grad
        # p.numel() returns the total number of scalar values in the tensor
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
