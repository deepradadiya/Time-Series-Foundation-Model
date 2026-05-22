"""Single transformer encoder layer with pre-norm architecture.

This module implements one layer of the PatchTST transformer encoder. It uses
the pre-norm pattern (LayerNorm before each sublayer) which improves training
stability compared to post-norm. The layer combines multi-head self-attention
with a position-wise feedforward network, each wrapped in a residual connection.

Related modules:
    - model/attention.py provides the MultiHeadSelfAttention used in this layer
    - model/encoder.py stacks multiple instances of this layer into the full encoder
    - config.py supplies D_MODEL, N_HEADS, D_FF, and DROPOUT hyperparameters
"""

import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer: LN → MHSA → Residual → LN → FFN → Residual.

    This layer processes patch embeddings through self-attention and a feedforward
    network. The pre-norm design applies LayerNorm *before* each sublayer rather
    than after, which helps gradient flow in deep networks.

    Architecture:
        1. LayerNorm → Multi-Head Self-Attention → Dropout → Add Residual
        2. LayerNorm → FFN (Linear → GELU → Linear) → Dropout → Add Residual

    Args:
        d_model: Dimension of input and output embeddings (default 256).
        n_heads: Number of attention heads (default 8).
        d_ff: Hidden dimension of the feedforward network (default 1024 = 4 * d_model).
        dropout: Dropout probability applied after attention and FFN (default 0.1).

    Example:
        >>> layer = TransformerEncoderLayer(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        >>> x = torch.randn(4, 63, 256)  # (batch, num_patches, d_model)
        >>> out = layer(x)  # same shape: (4, 63, 256)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the transformer encoder layer components.

        Args:
            d_model: Embedding dimension for inputs, attention, and outputs.
            n_heads: Number of parallel attention heads.
            d_ff: Inner dimension of the two-layer feedforward network.
            dropout: Dropout rate for regularization after sublayers.
        """
        super().__init__()

        # --- Layer Normalization ---
        # Pre-norm: normalize inputs before each sublayer for stable training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Multi-Head Self-Attention sublayer ---
        # Allows the model to attend to different positions simultaneously
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Feedforward Network (FFN) sublayer ---
        # Two linear transformations with GELU activation in between.
        # Expands from d_model (256) to d_ff (1024), then projects back to d_model.
        self.ffn = nn.Sequential(
            # First linear layer: expand dimension (256 → 1024)
            nn.Linear(d_model, d_ff),
            # GELU activation: smooth approximation of ReLU, standard in transformers
            nn.GELU(),
            # Second linear layer: project back to model dimension (1024 → 256)
            nn.Linear(d_ff, d_model),
        )

        # --- Dropout layers ---
        # Applied after attention output and after FFN output for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the pre-norm transformer encoder layer.

        The forward pass applies two sublayers with residual connections:
            1. Pre-norm → Self-Attention → Dropout → Residual addition
            2. Pre-norm → FFN → Dropout → Residual addition

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
               For PatchTST, seq_len = 63 (number of patches) and d_model = 256.

        Returns:
            Output tensor of the same shape (batch_size, seq_len, d_model),
            with contextual information from self-attention and nonlinear
            transformation from the feedforward network.
        """
        # --- Sublayer 1: Multi-Head Self-Attention with residual ---
        # Apply layer norm before attention (pre-norm architecture)
        normed = self.norm1(x)
        # Compute self-attention over all patch positions
        attn_output = self.attention(normed)
        # Apply dropout after attention for regularization
        attn_output = self.dropout1(attn_output)
        # Add residual connection: preserves original signal and aids gradient flow
        x = x + attn_output

        # --- Sublayer 2: Feedforward Network with residual ---
        # Apply layer norm before FFN (pre-norm architecture)
        normed = self.norm2(x)
        # Pass through FFN: 256 → 1024 (GELU) → 256
        ffn_output = self.ffn(normed)
        # Apply dropout after FFN for regularization
        ffn_output = self.dropout2(ffn_output)
        # Add residual connection: combines FFN output with attention output
        x = x + ffn_output

        return x
