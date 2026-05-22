"""Property-based and unit tests for the PatchTST assembly module.

Tests validate:
- Property 11: Full model end-to-end shape (batch, 512) → (batch, 96, 3)
- Property 12: forecast() returns dict with correct keys and tensor shapes
- Unit test: Parameter count within 8-12M budget

Uses Hypothesis for property-based testing with random batch sizes.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from model.patch_tst import PatchTST


# Instantiate model once for all tests (expensive to create repeatedly)
_model = PatchTST()
_model.eval()


class TestPatchTSTEndToEndShape:
    """Property 11: Full Model End-to-End Shape.

    *For any* input of shape (batch, 512), the full PatchTST model SHALL
    produce output of shape (batch, 96, 3).

    **Validates: Requirements 5.2**
    """

    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=10, deadline=None)
    def test_output_shape(self, batch_size: int) -> None:
        """For input (batch, 512), verify output shape is (batch, 96, 3)."""
        x = torch.randn(batch_size, 512)

        with torch.no_grad():
            output = _model(x)

        assert output.shape == (batch_size, 96, 3), (
            f"Expected shape ({batch_size}, 96, 3), got {output.shape}"
        )


class TestPatchTSTForecastMethod:
    """Property 12: Forecast Method Returns Correct Structure.

    *For any* valid input, the forecast() method SHALL return a dictionary
    with keys 'p10', 'p50', 'p90', each containing a tensor of shape
    (batch, forecast_horizon).

    **Validates: Requirements 5.3, 5.6**
    """

    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=10, deadline=None)
    def test_forecast_returns_correct_structure(self, batch_size: int) -> None:
        """Verify forecast() returns dict with keys 'p10', 'p50', 'p90' and correct shapes."""
        x = torch.randn(batch_size, 512)

        with torch.no_grad():
            result = _model.forecast(x, mean=0.5, std=1.2)

        # Verify result is a dict with expected keys
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        expected_keys = {"p10", "p50", "p90"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

        # Verify each value has correct shape (batch, forecast_horizon=96)
        for key in expected_keys:
            assert result[key].shape == (batch_size, 96), (
                f"Expected {key} shape ({batch_size}, 96), got {result[key].shape}"
            )


class TestPatchTSTParameterCount:
    """Unit test: Parameter count within 8-12M budget.

    Verify that the default PatchTST configuration produces a model with
    between 8 million and 12 million trainable parameters.

    **Validates: Requirements 7.1, 7.2, 7.3**
    """

    def test_parameter_count_within_budget(self) -> None:
        """Verify parameter count is between 8M and 12M with default config."""
        count = _model.count_parameters()

        assert 8_000_000 <= count <= 12_000_000, (
            f"Parameter count {count:,} is outside the target range of "
            f"8,000,000 to 12,000,000"
        )
