"""Patch embedding module: segment raw time series into overlapping patches and project to embedding space.

This module implements the first stage of the PatchTST pipeline. It takes a raw
univariate time series and converts it into a sequence of embedded patch "tokens"
that the transformer encoder can process. The two operations are:
1. Patch extraction using torch.Tensor.unfold() — efficient sliding window
2. Linear projection mapping each patch (length 16) to a dense vector (dim 256)

The patches become the "tokens" for the transformer, analogous to word tokens in
NLP. Each patch captures a local temporal pattern, and the transformer learns
relationships between patches (cross-patch attention captures seasonality, trends).

================================================================================
ASCII-ART: How Patching Works
================================================================================

Raw time series (length 512):
[──────────────────────────────────────────────────────────────────────────────]
 t=0                                                                       t=511

Patch extraction (patch_len=16, stride=8):
Patch 0:  [████████████████]                                    (t=0   to t=15)
Patch 1:          [████████████████]                            (t=8   to t=23)
Patch 2:                  [████████████████]                    (t=16  to t=31)
Patch 3:                          [████████████████]            (t=24  to t=39)
...
Patch 62:                                        [████████████████]  (t=496 to t=511)

Result: 63 patches, each becoming a "token" for the transformer.
Formula: num_patches = floor((seq_len - patch_len) / stride) + 1
                     = floor((512 - 16) / 8) + 1
                     = floor(496 / 8) + 1
                     = 62 + 1
                     = 63

Overlap between consecutive patches:
    patch_len - stride = 16 - 8 = 8 time steps of overlap
    This 50% overlap ensures no information is lost at patch boundaries
    and provides the transformer with redundant context for smoother attention.

After linear projection (patch_len=16 → d_model=256):
    Each 16-dimensional patch vector is mapped to a 256-dimensional embedding.
    (batch, 63, 16) → (batch, 63, 256)

================================================================================

Related modules:
    - config.py: Provides PATCH_LEN (16), PATCH_STRIDE (8), D_MODEL (256)
    - model/positional_encoding.py: Adds position info to the embedded patches
    - model/patch_tst.py: Assembles this module as the first stage of PatchTST
"""

import torch
import torch.nn as nn

from config import Config


class PatchEmbedding(nn.Module):
    """Segment a raw time series into overlapping patches and project to embedding space.

    This module combines patch extraction and linear embedding into a single step.
    Given a batch of univariate time series of shape (batch, seq_len), it:
    1. Uses torch.Tensor.unfold() to extract overlapping patches efficiently
    2. Applies a linear projection to map each patch to the model's hidden dimension

    The unfold operation is a vectorized sliding window — no Python loops needed.
    It produces patches with 50% overlap (stride=8, patch_len=16), ensuring smooth
    coverage of the input time series.

    Parameters:
        patch_len (int): Length of each patch in time steps. Default: 16.
        d_model (int): Embedding dimension for the transformer. Default: 256.
        stride (int): Step size between consecutive patch starts. Default: 8.

    Example:
        >>> embed = PatchEmbedding(patch_len=16, d_model=256, stride=8)
        >>> x = torch.randn(4, 512)          # batch of 4 time series, length 512
        >>> output = embed(x)                 # shape: (4, 63, 256)
        >>> output.shape
        torch.Size([4, 63, 256])
    """

    def __init__(
        self,
        patch_len: int = Config.PATCH_LEN,
        d_model: int = Config.D_MODEL,
        stride: int = Config.PATCH_STRIDE,
    ) -> None:
        """Initialize the patch embedding layer.

        Args:
            patch_len: Number of time steps in each patch (input dimension for projection).
            d_model: Output embedding dimension for the transformer.
            stride: Step size between the start of consecutive patches.
        """
        super().__init__()

        self.patch_len = patch_len
        self.d_model = d_model
        self.stride = stride

        # -----------------------------------------------------------------------
        # Linear Projection Layer
        # Maps each patch from raw time-step values (patch_len=16 dimensions) to
        # the transformer's hidden dimension (d_model=256). This is analogous to
        # the patch projection in Vision Transformers (ViT), where image patches
        # are linearly embedded before being fed to the transformer.
        #
        # Weight matrix shape: (patch_len, d_model) = (16, 256)
        # For each patch vector p of shape (16,):
        #   embedding = p @ W + b  →  shape (256,)
        # -----------------------------------------------------------------------
        self.projection = nn.Linear(
            in_features=patch_len,   # Input: raw patch values (16 time steps)
            out_features=d_model,    # Output: dense embedding vector (256 dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches and project to embedding space.

        Uses torch.Tensor.unfold() for efficient patch extraction without loops.
        The unfold operation creates a view of the input tensor with a sliding
        window, making it memory-efficient and fast on GPU.

        Args:
            x: Input tensor of shape (batch, seq_len) representing a batch of
                raw univariate time series.

        Returns:
            Embedded patches of shape (batch, num_patches, d_model) where
            num_patches = floor((seq_len - patch_len) / stride) + 1.

        Raises:
            ValueError: If seq_len < patch_len (cannot form even one patch).
        """
        # Get sequence length from input
        seq_len = x.shape[-1]

        # -----------------------------------------------------------------------
        # Input Validation
        # The sequence must be at least as long as one patch. Otherwise, unfold
        # would produce zero patches and downstream modules would fail silently.
        # -----------------------------------------------------------------------
        if seq_len < self.patch_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) must be >= patch_len "
                f"({self.patch_len}) to form at least one complete patch."
            )

        # -----------------------------------------------------------------------
        # Step 1: Patch Extraction via unfold()
        # torch.Tensor.unfold(dimension, size, step) creates a sliding window view.
        #   - dimension=1 (or -1): slide along the time axis
        #   - size=patch_len: each window captures patch_len time steps
        #   - step=stride: move the window by stride steps between patches
        #
        # Input shape:  (batch, seq_len)           e.g., (B, 512)
        # Output shape: (batch, num_patches, patch_len)  e.g., (B, 63, 16)
        #
        # This is equivalent to:
        #   patches[i] = x[:, i*stride : i*stride + patch_len]
        # but vectorized and without Python loops.
        # -----------------------------------------------------------------------
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches shape: (batch, num_patches, patch_len)

        # -----------------------------------------------------------------------
        # Step 2: Linear Projection
        # Apply the learned linear layer to each patch independently.
        # The nn.Linear layer broadcasts over batch and num_patches dimensions.
        #
        # Input shape:  (batch, num_patches, patch_len)   e.g., (B, 63, 16)
        # Output shape: (batch, num_patches, d_model)     e.g., (B, 63, 256)
        # -----------------------------------------------------------------------
        embedded = self.projection(patches)

        return embedded
