"""Unit tests for the evaluation/metrics.py module.

Tests cover MAE, MSE, MASE, and CRPS computation including edge cases
like perfect predictions, constant series, and multi-dimensional inputs.

Related modules:
    - evaluation/metrics.py: The module under test
    - config.py: Provides QUANTILES = [0.1, 0.5, 0.9]
"""

import numpy as np
import pytest

from evaluation.metrics import crps_quantile, mae, mase, mse


class TestMAE:
    """Tests for the mae function."""

    def test_known_value(self) -> None:
        """MAE of [1,2,3] vs [1.5,2.5,3.5] is 0.5."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        assert mae(preds, targets) == pytest.approx(0.5)

    def test_perfect_predictions(self) -> None:
        """MAE is zero when predictions equal targets."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert mae(data, data) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        """MAE is always non-negative."""
        preds = np.random.randn(100)
        targets = np.random.randn(100)
        assert mae(preds, targets) >= 0.0

    def test_symmetric(self) -> None:
        """MAE(a, b) == MAE(b, a) since |a-b| == |b-a|."""
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 1.0, 4.0])
        assert mae(a, b) == pytest.approx(mae(b, a))

    def test_multidimensional(self) -> None:
        """MAE works on 2D arrays (n_windows, horizon)."""
        preds = np.ones((5, 96))
        targets = np.zeros((5, 96))
        assert mae(preds, targets) == pytest.approx(1.0)


class TestMSE:
    """Tests for the mse function."""

    def test_known_value(self) -> None:
        """MSE of [1,2,3] vs [1.5,2.5,3.5] is 0.25."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        assert mse(preds, targets) == pytest.approx(0.25)

    def test_perfect_predictions(self) -> None:
        """MSE is zero when predictions equal targets."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert mse(data, data) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        """MSE is always non-negative."""
        preds = np.random.randn(100)
        targets = np.random.randn(100)
        assert mse(preds, targets) >= 0.0

    def test_mse_greater_than_or_equal_mae_squared(self) -> None:
        """MSE >= MAE^2 by Jensen's inequality."""
        preds = np.array([1.0, 5.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        assert mse(preds, targets) >= mae(preds, targets) ** 2


class TestMASE:
    """Tests for the mase function."""

    def test_known_value(self) -> None:
        """MASE with known seasonal naive errors."""
        # targets: [1, 3, 2, 5, 4, 6], period=2
        # naive errors: |2-1|=1, |5-3|=2, |4-2|=2, |6-5|=1 -> mean=1.5
        # preds offset by 0.5 -> MAE=0.5
        # MASE = 0.5 / 1.5 = 0.3333
        targets = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
        preds = targets + 0.5
        result = mase(preds, targets, seasonal_period=2)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_perfect_predictions(self) -> None:
        """MASE is zero when predictions equal targets (non-periodic)."""
        targets = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
        result = mase(targets, targets, seasonal_period=2)
        assert result == pytest.approx(0.0)

    def test_periodic_series_returns_inf(self) -> None:
        """MASE returns inf for perfectly periodic series (zero naive error)."""
        targets = np.tile([1.0, 2.0, 3.0], 50)
        preds = targets + 0.1
        result = mase(preds, targets, seasonal_period=3)
        assert result == float("inf")

    def test_default_seasonal_period_is_24(self) -> None:
        """Default seasonal_period is 24 (hourly data with daily cycle)."""
        np.random.seed(0)
        targets = np.random.randn(200)
        preds = targets + np.random.randn(200) * 0.1
        # Just verify it runs without error with default period
        result = mase(preds, targets)
        assert result >= 0.0

    def test_multidimensional_flattened(self) -> None:
        """MASE flattens multi-dimensional inputs before computing."""
        targets = np.random.randn(5, 96)
        preds = targets + 0.5
        # Should work without error
        result = mase(preds, targets, seasonal_period=24)
        assert result >= 0.0


class TestCRPSQuantile:
    """Tests for the crps_quantile function."""

    def test_perfect_predictions_zero_crps(self) -> None:
        """CRPS is zero when all quantiles equal the target."""
        targets = np.array([2.0, 5.0])
        # All quantiles predict exactly the target
        q_preds = np.stack([targets, targets, targets], axis=-1)
        result = crps_quantile(q_preds, targets, [0.1, 0.5, 0.9])
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self) -> None:
        """CRPS is always non-negative."""
        np.random.seed(42)
        q_preds = np.sort(np.random.randn(10, 96, 3), axis=-1)
        targets = np.random.randn(10, 96)
        result = crps_quantile(q_preds, targets, [0.1, 0.5, 0.9])
        assert result >= 0.0

    def test_wider_intervals_higher_crps(self) -> None:
        """Wider prediction intervals generally produce higher CRPS."""
        targets = np.zeros(100)
        # Narrow intervals
        narrow = np.stack([
            targets - 0.1,
            targets,
            targets + 0.1,
        ], axis=-1)
        # Wide intervals
        wide = np.stack([
            targets - 10.0,
            targets,
            targets + 10.0,
        ], axis=-1)
        crps_narrow = crps_quantile(narrow, targets, [0.1, 0.5, 0.9])
        crps_wide = crps_quantile(wide, targets, [0.1, 0.5, 0.9])
        assert crps_wide > crps_narrow

    def test_3d_input_shape(self) -> None:
        """CRPS works with (n_windows, horizon, 3) shaped predictions."""
        q_preds = np.random.randn(5, 96, 3)
        q_preds.sort(axis=-1)  # Ensure monotonicity
        targets = np.random.randn(5, 96)
        # Should run without error
        result = crps_quantile(q_preds, targets, [0.1, 0.5, 0.9])
        assert isinstance(result, float)

    def test_single_quantile(self) -> None:
        """CRPS works with a single quantile level."""
        targets = np.array([1.0, 2.0, 3.0])
        q_preds = np.array([[1.5], [2.5], [3.5]])
        result = crps_quantile(q_preds, targets, [0.5])
        # With tau=0.5, pinball = 0.5 * |error| for both over/under
        # errors are all -0.5, so pinball = 0.5 * 0.5 = 0.25 each
        # CRPS = (2/1) * 0.25 = 0.5
        assert result == pytest.approx(0.5)
