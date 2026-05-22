"""Full PatchTST encoder: stack of N transformer layers with final layer norm.

This module stacks multiple TransformerEncoderLayer instances into the complete
encoder used by the PatchTST model. It applies a final LayerNorm after the last
layer to stabilize the output representations before they are passed to downstream
heads (reconstruction head for pretraining, probabilistic head for forecasting).

Related modules:
    - model/transformer_layer.py provides the individual TransformerEncoderLayer
    - model/patchtst.py uses this encoder as the core processing component
    - config.py supplies N_LAYERS, D_MODEL, N_HEADS, D_FF, and DROPOUT
"""

import torch
import torch.nn as nn

from config import Config
from model.transformer_layer import TransformerEncoderLayer


class PatchTSTEncoder(nn.Module):
    """Stack of N transformer encoder layers with a final layer normalization.

    This encoder processes patch embeddings through multiple self-attention and
    feedforward layers, building increasingly rich contextual representations.
    The final LayerNorm ensures stable output magnitudes regardless of depth.

    Architecture:
        Input → [TransformerEncoderLayer × N_LAYERS] → LayerNorm → Output

    Each layer applies pre-norm self-attention and feedforward processing with
    residual connections, allowing the model to learn hierarchical temporal
    patterns across the patch sequence.

    Args:
        n_layers: Number of transformer encoder layers to stack (default 6).
        d_model: Dimension of input/output embeddings (default 256).
        n_heads: Number of attention heads per layer (default 8).
        d_ff: Hidden dimension of feedforward networks (default 1024).
        dropout: Dropout probability for regularization (default 0.1).

    Example:
        >>> encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024)
        >>> x = torch.randn(4, 63, 256)  # (batch, num_patches, d_model)
        >>> out = encoder(x)  # same shape: (4, 63, 256)
    """

    def __init__(
        self,
        n_layers: int = Config.N_LAYERS,
        d_model: int = Config.D_MODEL,
        n_heads: int = Config.N_HEADS,
        d_ff: int = Config.D_FF,
        dropout: float = Config.DROPOUT,
    ) -> None:
        """Initialize the encoder with N stacked transformer layers and final norm.

        Args:
            n_layers: Number of transformer encoder layers (depth of the encoder).
            d_model: Embedding dimension throughout the encoder.
            n_heads: Number of parallel attention heads in each layer.
            d_ff: Inner dimension of the feedforward network in each layer.
            dropout: Dropout rate applied within each layer.
        """
        super().__init__()

        # Store configuration for reference
        self.n_layers = n_layers
        self.d_model = d_model

        # -----------------------------------------------------------------------
        # Transformer Encoder Layers
        # We use nn.ModuleList to register each layer as a submodule so that
        # PyTorch properly tracks parameters, handles device placement, and
        # includes them in state_dict for checkpointing.
        # Each layer is an independent TransformerEncoderLayer with its own
        # learned weights for attention and feedforward processing.
        # -----------------------------------------------------------------------
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # -----------------------------------------------------------------------
        # Final Layer Normalization
        # Applied after the last transformer layer to normalize the output
        # representations. This is standard practice in pre-norm transformers
        # because the residual stream accumulates unnormalized values through
        # the layers. The final norm ensures consistent output magnitudes for
        # downstream heads (reconstruction or forecasting).
        # -----------------------------------------------------------------------
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through all encoder layers and apply final normalization.

        Each layer processes the input sequentially, building richer contextual
        representations through self-attention and feedforward transformations.
        The final LayerNorm stabilizes the output for downstream consumption.

        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model).
               Typically (B, 63, 256) for PatchTST with default configuration.

        Returns:
            Output tensor of shape (batch_size, num_patches, d_model) containing
            contextualized patch representations ready for task-specific heads.
        """
        # Pass through each transformer layer sequentially
        # Each layer refines the representations with self-attention and FFN
        for layer in self.layers:
            x = layer(x)

        # Apply final layer normalization to stabilize output magnitudes
        x = self.final_norm(x)

        return x
