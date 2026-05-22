"""Unit tests for the data/preprocess.py module.

Tests cover normalization (z-score), inverse normalization, chronological splitting,
short series filtering, zero-std handling, and JSON stats persistence.

Related modules:
    - data/preprocess.py: The module under test
    - config.py: Provides TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CONTEXT_LENGTH
"""

import json
import os
import tempfile
import warnings

import numpy as np
import pytest

from data.preprocess import (
    compute_normalization_stats,
    filter_short_series,
    inverse_normalize,
    load_normalization_stats,
    normalize,
    preprocess_dataset,
    save_normalization_stats,
    split_chronological,
)


class TestComputeNormalizationStats:
    """Tests for compute_normalization_stats function."""

    def test_correct_mean_and_std(self) -> None:
        """Stats match numpy's mean and std for normal data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        stats = compute_normalization_stats(data)
        np.testing.assert_allclose(stats["mean"], [3.0, 4.0])
        np.testing.assert_allclose(stats["std"], np.std(data, axis=0).tolist())

    def test_zero_std_channel_set_to_one(self) -> None:
        """Constant channel gets std=1.0 to avoid division by zero."""
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stats = compute_normalization_stats(data)
            # Channel 0 is constant, should trigger warning
            assert any("Channel 0" in str(warning.message) for warning in w)
        assert stats["std"][0] == 1.0

    def test_1d_input_handled(self) -> None:
        """1D array is treated as single-channel data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_normalization_stats(data)
        assert len(stats["mean"]) == 1
        assert len(stats["std"]) == 1
        np.testing.assert_allclose(stats["mean"][0], 3.0)

    def test_returns_lists_not_arrays(self) -> None:
        """Stats values are Python lists (JSON-serializable)."""
        data = np.random.randn(50, 3)
        stats = compute_normalization_stats(data)
        assert isinstance(stats["mean"], list)
        assert isinstance(stats["std"], list)


class TestNormalize:
    """Tests for the normalize function."""

    def test_zero_mean_unit_variance(self) -> None:
        """Normalized training data has approximately zero mean and unit std."""
        data = np.random.randn(1000, 2) * 5 + 10
        stats = compute_normalization_stats(data)
        normalized = normalize(data, stats)
        np.testing.assert_allclose(np.mean(normalized, axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(np.std(normalized, axis=0), [1, 1], atol=1e-10)

    def test_1d_input_output(self) -> None:
        """1D input produces 1D output."""
        data = np.array([10.0, 20.0, 30.0])
        stats = {"mean": [20.0], "std": [10.0]}
        result = normalize(data, stats)
        assert result.ndim == 1
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])


class TestInverseNormalize:
    """Tests for the inverse_normalize function."""

    def test_round_trip(self) -> None:
        """normalize then inverse_normalize recovers original values."""
        data = np.random.randn(100, 4) * 3 + 7
        stats = compute_normalization_stats(data)
        recovered = inverse_normalize(normalize(data, stats), stats)
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_1d_round_trip(self) -> None:
        """Round-trip works for 1D arrays."""
        data = np.array([1.5, 2.5, 3.5, 4.5])
        stats = compute_normalization_stats(data)
        recovered = inverse_normalize(normalize(data, stats), stats)
        np.testing.assert_allclose(recovered, data, atol=1e-10)


class TestSplitChronological:
    """Tests for the split_chronological function."""

    def test_default_ratios_100_steps(self) -> None:
        """100 steps split into 70/15/15."""
        data = np.arange(100)
        train, val, test = split_chronological(data)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_preserves_order(self) -> None:
        """Splits are contiguous and in chronological order."""
        data = np.arange(200).reshape(-1, 1)
        train, val, test = split_chronological(data)
        # Train comes first, then val, then test
        assert train[-1] < val[0]
        assert val[-1] < test[0]

    def test_concatenation_equals_original(self) -> None:
        """Concatenating splits recovers the original array."""
        data = np.random.randn(1000, 3)
        train, val, test = split_chronological(data)
        reconstructed = np.concatenate([train, val, test], axis=0)
        np.testing.assert_array_equal(reconstructed, data)

    def test_no_overlap(self) -> None:
        """Splits do not share any indices."""
        data = np.arange(50)
        train, val, test = split_chronological(data)
        all_values = np.concatenate([train, val, test])
        assert len(all_values) == len(data)
        assert len(set(all_values.tolist())) == len(data)

    def test_custom_ratios(self) -> None:
        """Custom split ratios are respected."""
        data = np.arange(100)
        train, val, test = split_chronological(data, 0.8, 0.1, 0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestFilterShortSeries:
    """Tests for the filter_short_series function."""

    def test_filters_short_series(self) -> None:
        """Series below min_length are removed."""
        series_list = [np.zeros(512), np.zeros(100), np.zeros(1000)]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = filter_short_series(series_list, min_length=512)
        assert len(result) == 2

    def test_preserves_order(self) -> None:
        """Retained series maintain their original order."""
        s1 = np.ones(600)
        s2 = np.ones(50)
        s3 = np.ones(700) * 2
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = filter_short_series([s1, s2, s3], min_length=512)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], s1)
        np.testing.assert_array_equal(result[1], s3)

    def test_warns_on_discard(self) -> None:
        """A UserWarning is issued for each discarded series."""
        series_list = [np.zeros(100), np.zeros(200)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            filter_short_series(series_list, min_length=512)
            assert len(w) == 2
            assert all(issubclass(warning.category, UserWarning) for warning in w)

    def test_empty_input(self) -> None:
        """Empty list returns empty list."""
        result = filter_short_series([], min_length=512)
        assert result == []

    def test_all_valid(self) -> None:
        """If all series are long enough, all are retained."""
        series_list = [np.zeros(512), np.zeros(1024)]
        result = filter_short_series(series_list, min_length=512)
        assert len(result) == 2

    def test_series_ids_in_warning(self) -> None:
        """Custom series IDs appear in warning messages."""
        series_list = [np.zeros(100)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            filter_short_series(series_list, min_length=512, series_ids=["my_series"])
            assert "my_series" in str(w[0].message)


class TestSaveLoadNormalizationStats:
    """Tests for save and load normalization stats."""

    def test_save_creates_json_file(self) -> None:
        """save_normalization_stats creates a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = {"mean": [1.0, 2.0], "std": [0.5, 1.5]}
            path = save_normalization_stats(stats, "energy", tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "energy" in data
            assert data["energy"] == stats

    def test_load_recovers_saved_stats(self) -> None:
        """load_normalization_stats returns the same dict that was saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = {"mean": [3.14, 2.71], "std": [1.0, 0.5]}
            save_normalization_stats(stats, "weather", tmpdir)
            loaded = load_normalization_stats("weather", tmpdir)
            assert loaded == stats

    def test_load_missing_file_raises(self) -> None:
        """Loading from non-existent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_normalization_stats("nonexistent", tmpdir)


class TestPreprocessDataset:
    """Tests for the full preprocess_dataset pipeline."""

    def test_output_shapes(self) -> None:
        """Pipeline produces correctly shaped train/val/test arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = np.random.randn(1000, 3)
            train, val, test, stats = preprocess_dataset(raw, "test", tmpdir)
            assert train.shape == (700, 3)
            assert val.shape == (150, 3)
            assert test.shape == (150, 3)

    def test_stats_file_created(self) -> None:
        """Pipeline saves normalization stats JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = np.random.randn(500, 2)
            preprocess_dataset(raw, "finance", tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "finance_norm_stats.json"))

    def test_normalized_train_has_zero_mean(self) -> None:
        """Training split after normalization has approximately zero mean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = np.random.randn(2000, 2) * 10 + 50
            train, _, _, _ = preprocess_dataset(raw, "test", tmpdir)
            np.testing.assert_allclose(
                np.mean(train, axis=0), [0, 0], atol=1e-10
            )
