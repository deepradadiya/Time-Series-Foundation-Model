"""Unit tests for the reconstruction head module.

Tests the ReconstructionHead linear projection and the masked MSE loss
computation function to ensure correct shapes, loss behavior, and gradient flow.
"""

import torch
import pytest

from pretraining.reconstruction_head import (
    ReconstructionHead,
    compute_masked_reconstruction_loss,
)
from config import Config


class TestReconstructionHead:
    """Tests for the ReconstructionHead nn.Module."""

    def test_output_shape_standard(self) -> None:
        """Output shape should be (batch, num_patches, patch_len) for standard input."""
        head = ReconstructionHead(d_model=256, patch_len=16)
        encoder_out = torch.randn(4, 63, 256)
        output = head(encoder_out)
        assert output.shape == (4, 63, 16)

    def test_output_shape_single_sample(self) -> None:
        """Should work with batch size of 1."""
        head = ReconstructionHead(d_model=256, patch_len=16)
        encoder_out = torch.randn(1, 63, 256)
        output = head(encoder_out)
        assert output.shape == (1, 63, 16)

    def test_uses_config_defaults(self) -> None:
        """Default parameters should match Config values."""
        head = ReconstructionHead()
        assert head.d_model == Config.D_MODEL
        assert head.patch_len == Config.PATCH_LEN

    def test_parameter_count(self) -> None:
        """Linear layer should have d_model * patch_len + patch_len parameters."""
        head = ReconstructionHead(d_model=256, patch_len=16)
        # Linear(256, 16) has 256*16 weights + 16 bias = 4112 parameters
        total_params = sum(p.numel() for p in head.parameters())
        assert total_params == 256 * 16 + 16

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the head back to the input."""
        head = ReconstructionHead(d_model=256, patch_len=16)
        encoder_out = torch.randn(2, 63, 256, requires_grad=True)
        output = head(encoder_out)
        loss = output.sum()
        loss.backward()
        assert encoder_out.grad is not None


class TestMaskedReconstructionLoss:
    """Tests for the compute_masked_reconstruction_loss function."""

    def test_zero_mask_returns_zero(self) -> None:
        """Loss should be 0.0 when no patches are masked."""
        predictions = torch.randn(4, 63, 16)
        targets = torch.randn(4, 63, 16)
        mask = torch.zeros(4, 63, dtype=torch.bool)
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        assert loss.item() == 0.0

    def test_full_mask_equals_mse(self) -> None:
        """When all patches are masked, loss should equal standard MSE."""
        predictions = torch.randn(4, 63, 16)
        targets = torch.randn(4, 63, 16)
        mask = torch.ones(4, 63, dtype=torch.bool)
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        expected_mse = ((predictions - targets) ** 2).mean()
        assert abs(loss.item() - expected_mse.item()) < 1e-5

    def test_unmasked_positions_ignored(self) -> None:
        """Large errors at unmasked positions should not affect the loss."""
        predictions = torch.zeros(2, 10, 16)
        targets = torch.zeros(2, 10, 16)
        # Large values at unmasked positions
        targets[:, 5:, :] = 100.0
        # Only mask first 4 patches where both are zero
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :4] = True
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        assert loss.item() == 0.0

    def test_loss_positive_for_different_values(self) -> None:
        """Loss should be positive when predictions differ from targets at masked positions."""
        predictions = torch.randn(4, 63, 16)
        targets = torch.randn(4, 63, 16)
        mask = torch.zeros(4, 63, dtype=torch.bool)
        mask[:, :25] = True  # ~40% masked
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        assert loss.item() > 0.0

    def test_perfect_reconstruction_zero_loss(self) -> None:
        """Loss should be zero when predictions exactly match targets at masked positions."""
        values = torch.randn(4, 63, 16)
        mask = torch.zeros(4, 63, dtype=torch.bool)
        mask[:, :25] = True
        loss = compute_masked_reconstruction_loss(values, values.clone(), mask)
        assert loss.item() < 1e-7

    def test_loss_is_differentiable(self) -> None:
        """Loss should support backpropagation."""
        head = ReconstructionHead(d_model=256, patch_len=16)
        encoder_out = torch.randn(2, 63, 256, requires_grad=True)
        predictions = head(encoder_out)
        targets = torch.randn(2, 63, 16)
        mask = torch.zeros(2, 63, dtype=torch.bool)
        mask[:, :25] = True
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        loss.backward()
        assert encoder_out.grad is not None

    def test_scalar_output(self) -> None:
        """Loss should be a scalar tensor (0-dimensional)."""
        predictions = torch.randn(4, 63, 16)
        targets = torch.randn(4, 63, 16)
        mask = torch.ones(4, 63, dtype=torch.bool)
        loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        assert loss.dim() == 0
