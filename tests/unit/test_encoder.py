"""Unit tests for the PatchTSTEncoder module.

Tests verify the full 6-layer encoder stack produces correct output shapes,
has the expected number of layers, applies final layer normalization, and
supports gradient flow for training.

Related modules:
    - model/encoder.py is the module under test
    - model/transformer_layer.py provides the individual layers stacked by the encoder
"""

import torch
import pytest

from model.encoder import PatchTSTEncoder


class TestPatchTSTEncoder:
    """Tests for the PatchTSTEncoder class."""

    def test_output_shape_default_config(self) -> None:
        """Output shape matches input shape with default PatchTST config."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        x = torch.randn(4, 63, 256)
        out = encoder(x)
        assert out.shape == (4, 63, 256)

    def test_output_shape_various_batch_sizes(self) -> None:
        """Encoder handles different batch sizes correctly."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 63, 256)
            out = encoder(x)
            assert out.shape == (batch_size, 63, 256)

    def test_output_shape_various_seq_lengths(self) -> None:
        """Encoder works with different sequence lengths (not just 63)."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        for seq_len in [1, 10, 63, 128]:
            x = torch.randn(2, seq_len, 256)
            out = encoder(x)
            assert out.shape == (2, seq_len, 256)

    def test_six_layers_stacked(self) -> None:
        """Encoder contains exactly 6 transformer layers by default."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        assert len(encoder.layers) == 6

    def test_custom_layer_count(self) -> None:
        """Encoder respects custom n_layers parameter."""
        for n_layers in [1, 3, 12]:
            encoder = PatchTSTEncoder(n_layers=n_layers, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
            assert len(encoder.layers) == n_layers

    def test_final_layer_norm_exists(self) -> None:
        """Encoder has a final LayerNorm applied after all layers."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        assert hasattr(encoder, 'final_norm')
        assert isinstance(encoder.final_norm, torch.nn.LayerNorm)
        assert encoder.final_norm.normalized_shape == (256,)

    def test_final_norm_applied(self) -> None:
        """Final LayerNorm actually normalizes the output (mean ~0, std ~1 per feature)."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        encoder.eval()
        x = torch.randn(4, 63, 256)
        out = encoder(x)
        # After LayerNorm, the last dimension should have mean ~0 and std ~1
        mean = out.mean(dim=-1)
        std = out.std(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1)

    def test_parameter_count(self) -> None:
        """Encoder parameter count is approximately 6 * single layer params + norm params."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        total_params = sum(p.numel() for p in encoder.parameters())
        # 6 layers * ~789K params each + final norm (256 weight + 256 bias) = ~4.74M
        assert 4_500_000 < total_params < 5_000_000

    def test_gradient_flow(self) -> None:
        """Gradients flow through all 6 layers back to the input."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        x = torch.randn(2, 63, 256, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, same input produces same output (no dropout randomness)."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        encoder.eval()
        x = torch.randn(2, 63, 256)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_train_mode_has_dropout_effect(self) -> None:
        """In train mode, dropout introduces stochasticity."""
        encoder = PatchTSTEncoder(n_layers=6, d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        encoder.train()
        x = torch.randn(2, 63, 256)
        out1 = encoder(x)
        out2 = encoder(x)
        # With dropout=0.1 across 6 layers, outputs should differ
        assert not torch.allclose(out1, out2)
