"""Unit tests for the top-level PatchTST model assembly.

Tests verify correct output shapes, parameter count constraints, input validation,
and channel-independent behavior of the full PatchTST model.
"""

import pytest
import torch

from config import Config
from model.patchtst import PatchTSTModel


class TestPatchTSTModel:
    """Tests for PatchTSTModel combining patch embedding and encoder."""

    def test_output_shape_standard_input(self) -> None:
        """Standard input (batch=4, 512) produces output (4, 63, 256)."""
        model = PatchTSTModel(Config)
        x = torch.randn(4, 512)
        out = model(x)
        assert out.shape == (4, 63, 256)

    def test_output_shape_single_batch(self) -> None:
        """Single sample input (1, 512) produces output (1, 63, 256)."""
        model = PatchTSTModel(Config)
        x = torch.randn(1, 512)
        out = model(x)
        assert out.shape == (1, 63, 256)

    def test_output_shape_various_batch_sizes(self) -> None:
        """Various batch sizes all produce correct output shapes."""
        model = PatchTSTModel(Config)
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 512)
            out = model(x)
            assert out.shape == (batch_size, 63, 256)

    def test_parameter_count_under_10m(self) -> None:
        """Total trainable parameters must be under 10 million."""
        model = PatchTSTModel(Config)
        param_count = model.count_parameters()
        assert param_count < 10_000_000
        # Also verify it's a reasonable number (not zero or trivially small)
        assert param_count > 100_000

    def test_raises_valueerror_for_short_input(self) -> None:
        """Inputs shorter than patch_len (16) raise ValueError."""
        model = PatchTSTModel(Config)
        short_input = torch.randn(2, 10)
        with pytest.raises(ValueError, match="minimum required length"):
            model(short_input)

    def test_raises_valueerror_for_length_one(self) -> None:
        """Input of length 1 raises ValueError."""
        model = PatchTSTModel(Config)
        x = torch.randn(1, 1)
        with pytest.raises(ValueError):
            model(x)

    def test_minimum_valid_input(self) -> None:
        """Input of exactly patch_len (16) produces one patch output."""
        model = PatchTSTModel(Config)
        x = torch.randn(2, 16)
        out = model(x)
        # floor((16 - 16) / 8) + 1 = 1 patch
        assert out.shape == (2, 1, 256)

    def test_gradient_flows_through_model(self) -> None:
        """Gradients flow from output back through all model parameters."""
        model = PatchTSTModel(Config)
        x = torch.randn(2, 512)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self) -> None:
        """Model in eval mode produces deterministic outputs."""
        model = PatchTSTModel(Config)
        model.eval()
        x = torch.randn(2, 512)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_output_dtype_float32(self) -> None:
        """Output tensor has float32 dtype matching input."""
        model = PatchTSTModel(Config)
        x = torch.randn(2, 512)
        out = model(x)
        assert out.dtype == torch.float32
