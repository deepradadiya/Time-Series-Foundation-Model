"""Sinusoidal Positional Encoding for PatchTST.

This module implements fixed (non-learnable) sinusoidal positional encoding from
"Attention is All You Need" (Vaswani et al., 2017). It adds position information
to patch embeddings so the transformer can distinguish patch order.

==============================================================================
SINUSOIDAL POSITIONAL ENCODING FORMULA
==============================================================================

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
    - pos   = position index (0, 1, 2, ..., seq_len-1)
    - i     = dimension index (0, 1, 2, ..., d_model/2 - 1)
    - d_model = embedding dimension (256)

The denominator 10000^(2i/d_model) creates wavelengths that form a geometric
progression from 2*pi to 10000*2*pi. This means:
    - Lower dimensions (small i) → high frequency → capture local position
    - Higher dimensions (large i) → low frequency → capture global position

==============================================================================
ASCII VISUALIZATION OF ENCODING PATTERN
==============================================================================

Encoding matrix (positions as rows, dimensions as columns):

         dim 0   dim 1   dim 2   dim 3   dim 4   dim 5  ...  dim d-1
         sin     cos     sin     cos     sin     cos         cos
        (fast)  (fast)  (med)   (med)   (slow)  (slow)      (slowest)
       ┌───────────────────────────────────────────────────────────────┐
pos 0  │  0.00    1.00    0.00    1.00    0.00    1.00  ...    1.00   │
pos 1  │  0.84    0.54    0.02    1.00    0.00    1.00  ...    1.00   │
pos 2  │  0.91   -0.42    0.03    1.00    0.00    1.00  ...    1.00   │
pos 3  │  0.14   -0.99    0.05    1.00    0.00    1.00  ...    1.00   │
pos 4  │ -0.76   -0.65    0.06    1.00    0.00    1.00  ...    1.00   │
  :    │   :       :       :       :       :       :            :     │
pos 62 │  0.74   -0.67    0.89    0.46    0.01    1.00  ...    1.00   │
       └───────────────────────────────────────────────────────────────┘

Key insight: Each position gets a UNIQUE encoding vector because the
combination of sin/cos at different frequencies creates a distinct fingerprint.
This is analogous to how radio stations use different frequencies to avoid
interference — each position "broadcasts" on all frequencies simultaneously.

Frequency pattern across dimensions:
    dim 0,1:   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (rapid oscillation — local patterns)
    dim 2,3:   ∿∿∿∿∿∿∿∿            (medium oscillation)
    dim 4,5:   ∿∿∿∿                 (slow oscillation)
      ...
    dim d-2,d-1: ∿                   (very slow — global position)

==============================================================================
"""

import math

import torch
import torch.nn as nn

from config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from "Attention is All You Need".

    Adds position-dependent sin/cos signals to input embeddings so the
    transformer can distinguish the order of patches. The encoding is
    pre-computed at initialization and registered as a buffer (not a
    parameter), meaning:
        - It does NOT consume optimizer memory
        - It does NOT receive gradients
        - It DOES persist across .to(device) calls

    Args:
        d_model: Embedding dimension (must be even). Default: 256.
        max_len: Maximum sequence length to pre-compute. Default: 128.
        dropout: Dropout probability applied after adding encoding. Default: 0.1.

    Input shape:  (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        max_len: int = 128,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Dropout applied after adding positional encoding
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the sinusoidal encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1) for broadcasting
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model)
        # Using log-space for numerical stability:
        #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model) for broadcasting over batch
        pe = pe.unsqueeze(0)

        # Register as a buffer — not a parameter, no gradients, moves with model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input and apply dropout.

        The encoding is sliced to match the actual sequence length of the input,
        allowing the module to handle variable-length sequences up to max_len.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional
            information added and dropout applied.
        """
        # Slice encoding to actual sequence length and add to input
        # pe shape: (1, max_len, d_model) → sliced to (1, seq_len, d_model)
        # Broadcasting adds encoding to every sample in the batch
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]

        # Apply dropout after adding positional encoding
        return self.dropout(x)
