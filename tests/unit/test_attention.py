"""Unit tests for the multi-head self-attention module.

Tests verify correct output shapes, parameter counts, dropout behavior,
and input validation for the MultiHeadSelfAttention class.
"""

import pytest
import torch

from model.attention import MultiHeadSelfAttention
from config import Config


class TestMultiHeadSelfAttention:
    """Tests for MultiHeadSelfAttention module."""

    def test_default_initialization(self) -> None:
        """Test that default config values are used correctly."""
        mhsa = MultiHeadSelfAttention()
        assert mhsa.d_model == 256
        assert mhsa.n_heads == 8
        assert mhsa.d_k == 32

    def test_output_shape_default_config(self) -> None:
        """Test output shape matches input shape with default PatchTST dimensions."""
        mhsa = MultiHeadSelfAttention()
        x = torch.randn(4, 63, 256)
        output = mhsa(x)
        assert output.shape == (4, 63, 256)

    def test_output_shape_single_batch(self) -> None:
        """Test output shape with batch size 1."""
        mhsa = MultiHeadSelfAttention()
        x = torch.randn(1, 63, 256)
        output = mhsa(x)
        assert output.shape == (1, 63, 256)

    def test_output_shape_varying_seq_len(self) -> None:
        """Test that attention works with different sequence lengths."""
        mhsa = MultiHeadSelfAttention()
        # Shorter sequence (e.g., fewer patches)
        x = torch.randn(2, 10, 256)
        output = mhsa(x)
        assert output.shape == (2, 10, 256)

    def test_custom_dimensions(self) -> None:
        """Test with custom d_model and n_heads."""
        mhsa = MultiHeadSelfAttention(d_model=128, n_heads=4, dropout=0.2)
        x = torch.randn(3, 20, 128)
        output = mhsa(x)
        assert output.shape == (3, 20, 128)
        assert mhsa.d_k == 32

    def test_invalid_d_model_raises_error(self) -> None:
        """Test that non-divisible d_model/n_heads raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadSelfAttention(d_model=255, n_heads=8)

    def test_dropout_disabled_in_eval_mode(self) -> None:
        """Test that outputs are deterministic in eval mode (no dropout)."""
        mhsa = MultiHeadSelfAttention()
        mhsa.eval()
        x = torch.randn(2, 63, 256)
        with torch.no_grad():
            out1 = mhsa(x)
            out2 = mhsa(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self) -> None:
        """Test that gradients flow through the attention module."""
        mhsa = MultiHeadSelfAttention()
        x = torch.randn(2, 63, 256, requires_grad=True)
        output = mhsa(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameter_count(self) -> None:
        """Test that parameter count is reasonable for the attention module."""
        mhsa = MultiHeadSelfAttention()
        total_params = sum(p.numel() for p in mhsa.parameters())
        # 4 linear layers: W_q, W_k, W_v, W_o each with weight (256x256) + bias (256)
        # = 4 * (256*256 + 256) = 4 * 65792 = 263168
        assert total_params == 4 * (256 * 256 + 256)
