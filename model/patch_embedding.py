"""Patch embedding module: linear projection and learnable positional encoding.

This module converts raw time series patches into dense vector representations that
the transformer encoder can process. It performs two operations:
1. A linear projection that maps each patch of length PATCH_LEN (16) to a vector
   of dimension D_MODEL (256).
2. Addition of learnable positional encodings so the transformer knows the ordering
   of patches within the sequence.

Related modules:
    - config.py provides PATCH_LEN, D_MODEL, and NUM_PATCHES constants
    - model/attention.py consumes the embedded patches for self-attention
    - model/patchtst.py assembles this module as the first stage of the full model
"""

import torch
import torch.nn as nn

from config import Config


class PatchEmbedding(nn.Module):
    """Linear projection of patches to D_MODEL dimension plus positional encoding.

    This module takes a batch of patch sequences where each patch is a raw segment
    of the time series (length 16), projects them into a higher-dimensional space
    (dimension 256), and adds learnable positional encodings so the model can
    distinguish patch positions within the sequence.

    The projection is a simple fully-connected (linear) layer applied independently
    to each patch. The positional encoding is a learnable parameter matrix — one
    vector per patch position — that is added element-wise to the projected patches.

    Parameters:
        patch_len (int): Length of each input patch (number of time steps per patch).
            Default is Config.PATCH_LEN = 16.
        d_model (int): Output embedding dimension for each patch.
            Default is Config.D_MODEL = 256.
        num_patches (int): Maximum number of patch positions supported.
            Default is Config.NUM_PATCHES = 63.

    Example:
        >>> embedding = PatchEmbedding(patch_len=16, d_model=256, num_patches=63)
        >>> patches = torch.randn(4, 63, 16)  # batch=4, 63 patches, each length 16
        >>> output = embedding(patches)        # shape: (4, 63, 256)
    """

    def __init__(
        self,
        patch_len: int = Config.PATCH_LEN,
        d_model: int = Config.D_MODEL,
        num_patches: int = Config.NUM_PATCHES,
    ) -> None:
        """Initialize the patch embedding layer and positional encodings.

        Args:
            patch_len: Number of time steps in each patch (input dimension).
            d_model: Embedding dimension for the transformer (output dimension).
            num_patches: Number of patch positions for positional encoding.
        """
        # Call the parent nn.Module constructor to register parameters properly
        super().__init__()

        # Store configuration values for use in forward pass and debugging
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_patches = num_patches

        # -----------------------------------------------------------------------
        # Linear Projection Layer
        # This layer maps each patch from its raw dimension (patch_len=16) to the
        # model's hidden dimension (d_model=256). It applies a learned weight matrix
        # W of shape (16, 256) and a bias vector b of shape (256,) to each patch:
        #   embedded_patch = patch @ W + b
        # This is analogous to the token embedding in NLP transformers, but instead
        # of looking up a discrete token in a vocabulary, we project a continuous
        # vector (the raw time series segment) into the embedding space.
        # -----------------------------------------------------------------------
        self.projection = nn.Linear(
            in_features=patch_len,   # Input: raw patch values (16 time steps)
            out_features=d_model,    # Output: dense embedding vector (256 dims)
        )

        # -----------------------------------------------------------------------
        # Learnable Positional Encoding
        # Unlike sinusoidal positional encodings (fixed), we use a learnable
        # parameter matrix of shape (num_patches, d_model) = (63, 256). Each row
        # corresponds to one patch position in the sequence. During training, the
        # model learns the optimal positional representation for each position.
        #
        # We use nn.Parameter so PyTorch tracks this tensor for gradient updates.
        # Initialization uses a normal distribution with small standard deviation
        # (0.02) to start with near-zero values, following common transformer
        # initialization practices (e.g., BERT, GPT-2).
        # -----------------------------------------------------------------------
        self.positional_encoding = nn.Parameter(
            torch.randn(num_patches, d_model) * 0.02
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Project patches to embedding space and add positional encodings.

        This method performs two sequential operations:
        1. Linear projection: each patch vector (length 16) is mapped to a dense
           vector (length 256) via a learned linear transformation.
        2. Position addition: a learnable positional encoding vector is added to
           each patch embedding based on its position in the sequence.

        Args:
            patches: Input tensor of shape (batch_size, num_patches, patch_len).
                Each element along the last dimension is a raw time step value
                from the original time series.

        Returns:
            Embedded patches of shape (batch_size, num_patches, d_model) with
            positional information encoded. Ready to be fed into the transformer
            encoder layers.

        Raises:
            RuntimeError: If the number of patches in the input exceeds the
                maximum number of positional encodings available.
        """
        # Get the actual number of patches in this input (may be <= num_patches)
        # This allows the module to handle inputs with fewer patches than the max
        seq_len = patches.shape[1]

        # -----------------------------------------------------------------------
        # Step 1: Linear Projection
        # Apply the learned linear layer to each patch independently.
        # Input shape:  (batch_size, seq_len, patch_len)  e.g., (B, 63, 16)
        # Output shape: (batch_size, seq_len, d_model)    e.g., (B, 63, 256)
        #
        # The nn.Linear layer broadcasts over the batch and sequence dimensions,
        # applying the same weights to every patch in every sample.
        # -----------------------------------------------------------------------
        embedded = self.projection(patches)

        # -----------------------------------------------------------------------
        # Step 2: Add Positional Encoding
        # Slice the positional encoding matrix to match the actual sequence length,
        # then add it to the projected embeddings. Broadcasting handles the batch
        # dimension automatically:
        #   embedded:             (batch_size, seq_len, d_model)
        #   positional_encoding:  (seq_len, d_model) → broadcasts to (1, seq_len, d_model)
        #
        # After addition, each patch embedding contains both content information
        # (from the linear projection) and position information (from the encoding).
        # This is critical because self-attention is permutation-invariant — without
        # positional encoding, the transformer cannot distinguish patch order.
        # -----------------------------------------------------------------------
        embedded = embedded + self.positional_encoding[:seq_len, :]

        return embedded
