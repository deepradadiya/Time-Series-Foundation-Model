"""Multi-head self-attention mechanism for the PatchTST transformer.

This module implements standard scaled dot-product multi-head self-attention.
Each attention head independently computes query-key-value projections and
attention weights, allowing the model to attend to information from different
representation subspaces at different positions simultaneously.

Related modules:
    - model/transformer_layer.py uses this attention module as a sublayer
    - model/encoder.py stacks layers that contain this attention mechanism
    - config.py supplies D_MODEL, N_HEADS, and DROPOUT hyperparameters
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with scaled dot-product.

    Splits the input embedding into multiple heads, computes attention
    independently in each head, then concatenates and projects the results.
    Uses scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V.

    Args:
        d_model: Total dimension of the model (split across heads). Default 256.
        n_heads: Number of parallel attention heads. Default 8.
        dropout: Dropout probability applied to attention weights. Default 0.1.

    Raises:
        ValueError: If d_model is not evenly divisible by n_heads.

    Example:
        >>> attn = MultiHeadSelfAttention(d_model=256, n_heads=8, dropout=0.1)
        >>> x = torch.randn(4, 63, 256)  # (batch, seq_len, d_model)
        >>> out = attn(x)  # same shape: (4, 63, 256)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize multi-head self-attention projections.

        Args:
            d_model: Embedding dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
        """
        super().__init__()

        # Validate that d_model splits evenly across heads
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        # Store configuration
        self.d_model = d_model
        self.n_heads = n_heads
        # Dimension per head: 256 / 8 = 32
        self.d_k = d_model // n_heads

        # --- Linear projections for Query, Key, Value ---
        # Each projects from d_model to d_model (all heads combined)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # --- Output projection ---
        # Combines multi-head outputs back to d_model dimension
        self.w_o = nn.Linear(d_model, d_model)

        # --- Attention dropout ---
        # Applied to attention weights before multiplying with values
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot-product attention: 1 / sqrt(d_k)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Steps:
            1. Project input to Q, K, V matrices
            2. Reshape to separate heads: (batch, heads, seq_len, d_k)
            3. Compute scaled dot-product attention per head
            4. Concatenate heads and apply output projection

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model) with
            attention-weighted contextual information.
        """
        batch_size, seq_len, _ = x.shape

        # --- Step 1: Linear projections for Q, K, V ---
        # Each has shape (batch, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # --- Step 2: Reshape to multi-head format ---
        # (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # --- Step 3: Scaled dot-product attention ---
        # Compute attention scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights (probabilities)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)

        # Multiply attention weights by values: (batch, heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # --- Step 4: Concatenate heads and project ---
        # (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final linear projection to combine head outputs
        output = self.w_o(attn_output)

        return output
