"""Property-based tests for the ProbabilisticHead module and quantile_loss function.

These tests validate the correctness properties defined in the design document
for the probabilistic forecasting head of the PatchTST architecture. Each test
uses Hypothesis to generate random inputs and verifies that universal properties hold.

Properties tested:
- Property 7 (Task 4.6): Monotonicity — P10 <= P50 <= P90 at every position
- Property 8 (Task 4.7): Output shape is (batch, forecast_horizon, 3)
- Property 9 (Task 4.8): Quantile loss returns non-negative scalar
- Property 10 (Task 4.9): Quantile loss returns zero when predictions equal targets
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from model.probabilistic_head import ProbabilisticHead, quantile_loss


# Default parameters matching the design spec
D_MODEL = 256
NUM_PATCHES = 63
FORECAST_HORIZON = 96
QUANTILES = [0.1, 0.5, 0.9]


class TestProbabilisticHeadMonotonicity:
    """**Validates: Requirements 4.5**

    Property 7: For any encoder output tensor, the ProbabilisticHead output
    SHALL satisfy P10 <= P50 <= P90 at every forecast timestep and batch element.
    """

    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=50)
    def test_p10_leq_p50_leq_p90(self, batch_size: int) -> None:
        """For random encoder outputs, verify output satisfies
        P10 <= P50 <= P90 at every position."""
        head = ProbabilisticHead(
            d_model=D_MODEL,
            num_patches=NUM_PATCHES,
            forecast_horizon=FORECAST_HORIZON,
            quantiles=QUANTILES,
        )
        head.eval()

        encoder_output = torch.randn(batch_size, NUM_PATCHES, D_MODEL)

        with torch.no_grad():
            output = head(encoder_output)

        # output shape: (batch, forecast_horizon, 3) where dim=-1 is [P10, P50, P90]
        p10 = output[:, :, 0]
        p50 = output[:, :, 1]
        p90 = output[:, :, 2]

        assert (p10 <= p50).all(), (
            f"Monotonicity violated: P10 > P50 found. "
            f"Max violation: {(p10 - p50).max().item()}"
        )
        assert (p50 <= p90).all(), (
            f"Monotonicity violated: P50 > P90 found. "
            f"Max violation: {(p50 - p90).max().item()}"
        )


class TestProbabilisticHeadOutputShape:
    """**Validates: Requirements 4.2**

    Property 8: For any encoder output of shape (batch, num_patches, d_model),
    the ProbabilisticHead SHALL produce output of shape (batch, forecast_horizon, 3).
    """

    @given(batch_size=st.integers(min_value=1, max_value=8))
    @settings(max_examples=50)
    def test_output_shape_is_correct(self, batch_size: int) -> None:
        """For random encoder outputs, verify output shape is
        (batch, forecast_horizon, 3)."""
        head = ProbabilisticHead(
            d_model=D_MODEL,
            num_patches=NUM_PATCHES,
            forecast_horizon=FORECAST_HORIZON,
            quantiles=QUANTILES,
        )
        head.eval()

        encoder_output = torch.randn(batch_size, NUM_PATCHES, D_MODEL)

        with torch.no_grad():
            output = head(encoder_output)

        expected_shape = (batch_size, FORECAST_HORIZON, 3)
        assert output.shape == expected_shape, (
            f"Expected output shape {expected_shape} but got {output.shape}"
        )


class TestQuantileLossNonNegativity:
    """**Validates: Requirements 6.5**

    Property 9: For any prediction and target tensor combination, the
    quantile_loss function SHALL return a non-negative scalar value.
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        horizon=st.integers(min_value=1, max_value=96),
    )
    @settings(max_examples=50)
    def test_loss_is_non_negative_scalar(self, batch_size: int, horizon: int) -> None:
        """For random predictions and targets, verify quantile_loss returns
        non-negative scalar."""
        predictions = torch.randn(batch_size, horizon, 3)
        targets = torch.randn(batch_size, horizon)

        loss = quantile_loss(predictions, targets, quantiles=QUANTILES)

        assert loss.dim() == 0, (
            f"Expected scalar (0-dim tensor) but got tensor with {loss.dim()} dims"
        )
        assert loss.item() >= 0, (
            f"Expected non-negative loss but got {loss.item()}"
        )


class TestQuantileLossZeroAtPerfectPrediction:
    """**Validates: Requirements 6.2**

    Property 10: For any target tensor, when predictions exactly equal targets
    for all quantiles, the quantile_loss SHALL return zero.
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        horizon=st.integers(min_value=1, max_value=96),
    )
    @settings(max_examples=50)
    def test_loss_is_zero_when_predictions_equal_targets(
        self, batch_size: int, horizon: int
    ) -> None:
        """When predictions equal targets, verify quantile_loss returns zero."""
        targets = torch.randn(batch_size, horizon)

        # Expand targets to match all 3 quantiles: (batch, horizon, 3)
        predictions = targets.unsqueeze(-1).expand(-1, -1, 3)

        loss = quantile_loss(predictions, targets, quantiles=QUANTILES)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), (
            f"Expected loss ~0 when predictions equal targets, but got {loss.item()}"
        )
