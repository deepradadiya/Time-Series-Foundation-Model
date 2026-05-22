"""Transformer Encoder for PatchTST time series forecasting.

This module implements the core transformer encoder stack that processes
patch embeddings through self-attention and feedforward layers. It learns
contextual relationships between patches — capturing temporal patterns like
seasonality, trends, and cross-patch dependencies.

Architecture Overview:
======================

Input: (batch, num_patches, d_model) = (B, 63, 256)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TransformerEncoderBlock (×6)                                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ LayerNorm → Multi-Head Self-Attention → Dropout → + Residual │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ LayerNorm → FFN (256→1024→256, GELU) → Dropout → + Residual  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
  Final LayerNorm
    │
    ▼
Output: (batch, num_patches, d_model) = (B, 63, 256)


Component Roles in Time Series Context:
========================================

1. SELF-ATTENTION (cross-patch pattern detection):
   - Each patch attends to ALL other patches in the sequence
   - Captures seasonal patterns: patch at t=0 can attend to patch at t=24h
     (daily seasonality) or t=168h (weekly seasonality)
   - Learns which historical patches are most relevant for forecasting

2. LAYER NORMALIZATION (training stability):
   - Pre-norm architecture: normalize BEFORE each sublayer
   - Prevents gradient explosion/vanishing in deep 6-layer stack
   - Stabilizes training dynamics, especially important for time series
     where input magnitudes can vary across different datasets

3. FEED-FORWARD NETWORK (independent patch processing):
   - Two linear layers with GELU activation: 256 → 1024 → 256
   - Processes each patch position independently (no cross-patch interaction)
   - Adds non-linear transformation capacity to refine representations
   - GELU is smoother than ReLU, standard in modern transformers (GPT, BERT)

4. RESIDUAL CONNECTIONS (gradient flow):
   - x + sublayer(x) around both attention and FFN
   - Ensures gradients flow directly through the 6-layer stack
   - Allows each layer to learn incremental refinements rather than
     full transformations — critical for training deep networks

Related modules:
    - model/patching.py provides input patch embeddings
    - model/positional_encoding.py adds position info before this encoder
    - model/probabilistic_head.py consumes encoder output for forecasting
    - config.py supplies D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# =============================================================================
# Multi-Head Self-Attention
# =============================================================================


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product.

    Splits the input embedding into multiple heads, computes attention
    independently in each head, then concatenates and projects the results.

    In the time series context, self-attention allows each patch to "look at"
    all other patches and learn which temporal positions are most informative.
    For example, a patch at hour 48 might strongly attend to the patch at
    hour 24 (capturing daily seasonality).

    Attention formula per head:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        - Q (queries): "What am I looking for?"
        - K (keys): "What do I contain?"
        - V (values): "What information do I provide?"
        - sqrt(d_k): Scaling factor to prevent softmax saturation

    Args:
        d_model: Total embedding dimension (split across heads). Default 256.
        n_heads: Number of parallel attention heads. Default 8.
        dropout: Dropout probability on attention weights. Default 0.1.

    Raises:
        ValueError: If d_model is not evenly divisible by n_heads.

    Example:
        >>> attn = MultiHeadSelfAttention(d_model=256, n_heads=8)
        >>> x = torch.randn(4, 63, 256)  # (batch, num_patches, d_model)
        >>> out = attn(x)  # (4, 63, 256) — same shape
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        n_heads: int = Config.N_HEADS,
        dropout: float = Config.DROPOUT,
    ) -> None:
        """Initialize Q/K/V projections and output projection.

        Args:
            d_model: Embedding dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.

        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        super().__init__()

        # --- Validation ---
        # d_model must split evenly across heads so each head gets d_k dimensions
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Each head needs an equal share of the embedding dimension."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        # Dimension per head: 256 / 8 = 32
        self.d_k = d_model // n_heads

        # --- Q/K/V Linear Projections ---
        # Each projects from d_model → d_model (all heads packed together)
        # After projection, we reshape to separate the heads
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection

        # --- Output Projection ---
        # After concatenating all heads, project back to d_model
        self.w_o = nn.Linear(d_model, d_model)

        # --- Attention Dropout ---
        # Applied to attention weights before multiplying with values
        # Randomly zeros out attention connections for regularization
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor: 1 / sqrt(d_k) prevents dot products from growing
        # too large in magnitude, which would push softmax into saturation
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Steps:
            1. Project input to Q, K, V matrices
            2. Reshape to separate heads: (batch, n_heads, seq_len, d_k)
            3. Compute scaled dot-product attention per head
            4. Concatenate heads and apply output projection

        Args:
            x: Input tensor of shape (batch, num_patches, d_model).

        Returns:
            Output tensor of shape (batch, num_patches, d_model) with
            attention-weighted contextual information from all patches.
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections — each (batch, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Step 2: Reshape to multi-head format
        # (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        # scores: (batch, n_heads, seq_len, seq_len)
        # Each entry scores[b, h, i, j] = how much patch i attends to patch j
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Softmax normalizes scores to attention weights (sum to 1 per query)
        attn_weights = F.softmax(scores, dim=-1)

        # Dropout on attention weights for regularization
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (batch, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # Step 4: Concatenate heads and project
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final output projection combines information from all heads
        output = self.w_o(attn_output)

        return output


# =============================================================================
# Transformer Encoder Block (Single Layer)
# =============================================================================


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture.

    Architecture (pre-norm — LayerNorm BEFORE each sublayer):

        x ─────────────────────────────────┐
        │                                  │ (residual)
        ▼                                  │
      LayerNorm                            │
        │                                  │
        ▼                                  │
      Multi-Head Self-Attention            │
        │                                  │
        ▼                                  │
      Dropout                              │
        │                                  │
        ▼                                  │
        + ◄────────────────────────────────┘
        │
        ├─────────────────────────────────┐
        │                                 │ (residual)
        ▼                                 │
      LayerNorm                           │
        │                                 │
        ▼                                 │
      FFN (Linear→GELU→Linear)            │
        │                                 │
        ▼                                 │
      Dropout                             │
        │                                 │
        ▼                                 │
        + ◄───────────────────────────────┘
        │
        ▼
      Output

    Pre-norm vs Post-norm:
        Pre-norm (used here) applies LayerNorm BEFORE the sublayer.
        This provides more stable gradients during training, especially
        important for deeper stacks (6 layers). Post-norm applies LayerNorm
        AFTER the residual addition and can be harder to train.

    Args:
        d_model: Embedding dimension. Default 256.
        n_heads: Number of attention heads. Default 8.
        d_ff: Feed-forward inner dimension. Default 1024.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        n_heads: int = Config.N_HEADS,
        d_ff: int = Config.D_FF,
        dropout: float = Config.DROPOUT,
    ) -> None:
        """Initialize encoder block with attention, FFN, norms, and dropout.

        Args:
            d_model: Embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension (typically 4 * d_model).
            dropout: Dropout probability for sublayer outputs.
        """
        super().__init__()

        # --- Layer Normalization (pre-norm architecture) ---
        # norm1: applied before self-attention
        # norm2: applied before feed-forward network
        # Normalizes across the d_model dimension for training stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Multi-Head Self-Attention ---
        # Allows each patch to attend to all other patches
        # Captures cross-patch temporal patterns (seasonality, trends)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        # --- Feed-Forward Network ---
        # Two linear layers with GELU activation
        # Processes each patch position independently (no cross-patch interaction)
        # Expands to d_ff=1024 for more capacity, then projects back to d_model=256
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 256 → 1024 (expand)
            nn.GELU(),                   # Smooth activation (better than ReLU)
            nn.Linear(d_ff, d_model),   # 1024 → 256 (compress back)
        )

        # --- Dropout ---
        # Applied after each sublayer (attention and FFN) before residual add
        # Regularizes by randomly dropping sublayer outputs during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one encoder block.

        Pre-norm architecture:
            1. LayerNorm → MHSA → Dropout → + Residual
            2. LayerNorm → FFN → Dropout → + Residual

        Args:
            x: Input tensor of shape (batch, num_patches, d_model).

        Returns:
            Output tensor of shape (batch, num_patches, d_model).
        """
        # --- Sublayer 1: Self-Attention with residual ---
        # Pre-norm: normalize before attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x  # Residual connection for gradient flow

        # --- Sublayer 2: Feed-Forward Network with residual ---
        # Pre-norm: normalize before FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x  # Residual connection for gradient flow

        return x


# =============================================================================
# Transformer Encoder (Full Stack)
# =============================================================================


class TransformerEncoder(nn.Module):
    """Stack of N transformer encoder blocks with final LayerNorm.

    Stacks multiple TransformerEncoderBlock instances sequentially.
    Each block refines the patch representations by attending to cross-patch
    patterns and applying non-linear transformations. The final LayerNorm
    stabilizes the output before passing to the probabilistic head.

    With 6 layers, the encoder progressively builds richer representations:
        - Early layers: capture local patterns (adjacent patch relationships)
        - Middle layers: capture medium-range dependencies (weekly patterns)
        - Later layers: capture global patterns (long-term trends, seasonality)

    Args:
        n_layers: Number of encoder blocks to stack. Default 6.
        d_model: Embedding dimension. Default 256.
        n_heads: Number of attention heads. Default 8.
        d_ff: Feed-forward inner dimension. Default 1024.
        dropout: Dropout probability. Default 0.1.

    Example:
        >>> encoder = TransformerEncoder(n_layers=6, d_model=256)
        >>> x = torch.randn(4, 63, 256)  # (batch, num_patches, d_model)
        >>> out = encoder(x)  # (4, 63, 256) — shape preserved
    """

    def __init__(
        self,
        n_layers: int = Config.N_LAYERS,
        d_model: int = Config.D_MODEL,
        n_heads: int = Config.N_HEADS,
        d_ff: int = Config.D_FF,
        dropout: float = Config.DROPOUT,
    ) -> None:
        """Initialize encoder stack with N blocks and final LayerNorm.

        Args:
            n_layers: Number of TransformerEncoderBlock layers.
            d_model: Embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        # --- Stack of Encoder Blocks ---
        # nn.ModuleList ensures all blocks are registered as submodules
        # (parameters are tracked for optimization and device transfers)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # --- Final LayerNorm ---
        # Applied after the last block to normalize the final output
        # This is standard in pre-norm architectures to ensure the output
        # is well-scaled before being consumed by downstream modules
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input sequentially through all encoder blocks + final norm.

        Each block refines the representation by:
            1. Attending to cross-patch patterns (self-attention)
            2. Applying non-linear transformations (FFN)

        Args:
            x: Input tensor of shape (batch, num_patches, d_model).

        Returns:
            Output tensor of shape (batch, num_patches, d_model) with
            contextualized patch representations ready for forecasting.
        """
        # Sequential pass through all encoder blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization for stable output
        x = self.final_norm(x)

        return x
