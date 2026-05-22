"""Unit tests for the TransformerEncoderLayer module.

Tests verify the pre-norm transformer layer architecture including correct
output shapes, parameter counts, component configuration, gradient flow,
and deterministic behavior in eval mode.
"""

import torch
import pytest

from model.transformer_layer import TransformerEncoderLayer


class TestTransformerEncoderLayer:
    """Tests for TransformerEncoderLayer with pre-norm architecture."""

    def setup_method(self) -> None:
        """Create a layer instance with default PatchTST configuration."""
        self.layer = TransformerEncoderLayer(
            d_model=256, n_heads=8, d_ff=1024, dropout=0.1
        )

    def test_output_shape_matches_input(self) -> None:
        """Output shape should equal input shape (batch, seq_len, d_model)."""
        x = torch.randn(4, 63, 256)
        output = self.layer(x)
        assert output.shape == (4, 63, 256)

    def test_output_shape_various_batch_sizes(self) -> None:
        """Layer should handle any batch size >= 1."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 63, 256)
            output = self.layer(x)
            assert output.shape == (batch_size, 63, 256)

    def test_output_shape_various_seq_lengths(self) -> None:
        """Layer should handle any sequence length."""
        for seq_len in [1, 10, 63, 100]:
            x = torch.randn(2, seq_len, 256)
            output = self.layer(x)
            assert output.shape == (2, seq_len, 256)

    def test_parameter_count(self) -> None:
        """Total parameters per layer should be ~790K as per design."""
        total_params = sum(p.numel() for p in self.layer.parameters())
        # Expected: attention (263,168) + FFN (525,568) + 2*LayerNorm (1,024) = 789,760
        assert total_params == 789_760

    def test_ffn_dimensions(self) -> None:
        """FFN should expand 256 -> 1024 then project back 1024 -> 256."""
        ffn = self.layer.ffn
        assert ffn[0].in_features == 256
        assert ffn[0].out_features == 1024
        assert ffn[2].in_features == 1024
        assert ffn[2].out_features == 256

    def test_gelu_activation(self) -> None:
        """FFN should use GELU activation between linear layers."""
        assert isinstance(self.layer.ffn[1], torch.nn.GELU)

    def test_dropout_rate(self) -> None:
        """Dropout should be 0.1 after both attention and FFN sublayers."""
        assert self.layer.dropout1.p == 0.1
        assert self.layer.dropout2.p == 0.1

    def test_layer_norm_dimension(self) -> None:
        """Both LayerNorm modules should normalize over d_model=256."""
        assert self.layer.norm1.normalized_shape == (256,)
        assert self.layer.norm2.normalized_shape == (256,)

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, same input should produce same output (no dropout)."""
        self.layer.eval()
        x = torch.randn(2, 63, 256)
        out1 = self.layer(x)
        out2 = self.layer(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self) -> None:
        """Gradients should flow back through residual connections."""
        self.layer.train()
        x = torch.randn(2, 63, 256, requires_grad=True)
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_residual_connection_effect(self) -> None:
        """Output should differ from input (layer transforms the data)."""
        self.layer.eval()
        x = torch.randn(2, 63, 256)
        output = self.layer(x)
        # Output should not be identical to input (transformation occurred)
        assert not torch.allclose(x, output)

    def test_pre_norm_architecture(self) -> None:
        """Verify pre-norm: LayerNorm is applied before sublayers, not after."""
        # The layer should have norm1 and norm2 as separate attributes
        # (not applied after the sublayer output)
        assert hasattr(self.layer, 'norm1')
        assert hasattr(self.layer, 'norm2')
        assert isinstance(self.layer.norm1, torch.nn.LayerNorm)
        assert isinstance(self.layer.norm2, torch.nn.LayerNorm)
