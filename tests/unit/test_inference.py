"""Unit tests for the zero-shot forecasting inference module.

Tests cover the sliding window computation, zero-shot forecast function,
error handling for short data, and the inverse normalization integration.
"""

import numpy as np
import pytest
import torch

from config import Config
from forecasting.inference import compute_num_windows, zero_shot_forecast
from forecasting.probabilistic_head import ProbabilisticForecastHead
from model.patchtst import PatchTSTModel


class TestComputeNumWindows:
    """Tests for the compute_num_windows helper function."""

    def test_standard_case(self) -> None:
        """Test window count for a typical data length."""
        # Data length 1000: (1000 - 512 - 96) / 96 + 1 = 392/96 + 1 = 4 + 1 = 5
        result = compute_num_windows(1000, 512, 96, 96)
        expected = (1000 - 512 - 96) // 96 + 1
        assert result == expected == 5

    def test_exact_minimum_one_window(self) -> None:
        """Test that exactly context + horizon gives 1 window."""
        # 512 + 96 = 608 is the minimum for one window
        result = compute_num_windows(608, 512, 96, 96)
        assert result == 1

    def test_too_short_returns_zero(self) -> None:
        """Test that data shorter than context + horizon gives 0 windows."""
        result = compute_num_windows(607, 512, 96, 96)
        assert result == 0

    def test_very_short_returns_zero(self) -> None:
        """Test that very short data gives 0 windows."""
        result = compute_num_windows(100, 512, 96, 96)
        assert result == 0

    def test_two_windows(self) -> None:
        """Test data length that produces exactly 2 windows."""
        # Need: 512 + 96 + 96 = 704 for 2 windows
        result = compute_num_windows(704, 512, 96, 96)
        assert result == 2

    def test_custom_stride(self) -> None:
        """Test with a custom stride value."""
        # stride=48 (overlapping): (1000 - 512 - 96) / 48 + 1 = 392/48 + 1 = 8 + 1 = 9
        result = compute_num_windows(1000, 512, 96, 48)
        expected = (1000 - 512 - 96) // 48 + 1
        assert result == expected

    def test_large_dataset(self) -> None:
        """Test with a large dataset (ETTh1-like size)."""
        # ETTh1 test split might be ~2500 time steps
        result = compute_num_windows(2500, 512, 96, 96)
        expected = (2500 - 512 - 96) // 96 + 1
        assert result == expected


class TestZeroShotForecast:
    """Tests for the zero_shot_forecast function."""

    @pytest.fixture
    def model_and_head(self) -> tuple[PatchTSTModel, ProbabilisticForecastHead]:
        """Create a model and head for testing."""
        model = PatchTSTModel(Config)
        head = ProbabilisticForecastHead(
            d_model=Config.D_MODEL,
            num_patches=Config.NUM_PATCHES,
            forecast_horizon=Config.FORECAST_HORIZON,
            quantiles=Config.QUANTILES,
        )
        return model, head

    def test_output_shape_single_window(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test output shape with data for exactly one window."""
        model, head = model_and_head
        data = np.random.randn(608).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        result = zero_shot_forecast(model, head, data, norm_stats)
        assert result.shape == (1, 96, 3)

    def test_output_shape_multiple_windows(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test output shape with data for multiple windows."""
        model, head = model_and_head
        data = np.random.randn(1000).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        result = zero_shot_forecast(model, head, data, norm_stats)
        expected_windows = compute_num_windows(1000)
        assert result.shape == (expected_windows, 96, 3)

    def test_raises_for_short_data(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test that ValueError is raised for data too short for one window."""
        model, head = model_and_head
        data = np.random.randn(500).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        with pytest.raises(ValueError, match="too short"):
            zero_shot_forecast(model, head, data, norm_stats)

    def test_no_gradient_computation(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test that no gradients are computed during inference."""
        model, head = model_and_head
        data = np.random.randn(700).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        # Run forecast
        zero_shot_forecast(model, head, data, norm_stats)

        # Verify no gradients accumulated on model parameters
        for param in model.parameters():
            assert param.grad is None
        for param in head.parameters():
            assert param.grad is None

    def test_eval_mode_set(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test that model and head are set to eval mode during inference."""
        model, head = model_and_head
        # Start in training mode
        model.train()
        head.train()

        data = np.random.randn(700).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        zero_shot_forecast(model, head, data, norm_stats)

        # After inference, model and head should be in eval mode
        assert not model.training
        assert not head.training

    def test_inverse_normalization_applied(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test that inverse normalization shifts predictions to original scale."""
        model, head = model_and_head
        data = np.random.randn(700).astype(np.float32)

        # With identity stats (mean=0, std=1), output should be raw model output
        identity_stats = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        result_identity = zero_shot_forecast(model, head, data, identity_stats)

        # With shifted stats (mean=100, std=1), output should be shifted by 100
        shifted_stats = {"mean": [100.0, 100.0, 100.0], "std": [1.0, 1.0, 1.0]}
        result_shifted = zero_shot_forecast(model, head, data, shifted_stats)

        # The shifted result should be approximately identity + 100
        np.testing.assert_allclose(
            result_shifted, result_identity + 100.0, atol=1e-4
        )

    def test_output_dtype_float(
        self, model_and_head: tuple[PatchTSTModel, ProbabilisticForecastHead]
    ) -> None:
        """Test that output is a float numpy array."""
        model, head = model_and_head
        data = np.random.randn(700).astype(np.float32)
        norm_stats = {"mean": [0.0], "std": [1.0]}

        result = zero_shot_forecast(model, head, data, norm_stats)
        assert result.dtype in (np.float32, np.float64)
