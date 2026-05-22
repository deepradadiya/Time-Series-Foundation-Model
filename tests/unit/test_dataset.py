"""Unit tests for data/dataset.py — TimeSeriesDataset and MultiDomainDataLoader.

Tests cover:
  - TimeSeriesDataset sample count, output shapes, and sliding window correctness
  - MultiDomainDataLoader round-robin ordering and epoch termination
  - Edge cases: short series, single-sample datasets, multivariate data
"""

import numpy as np
import pytest
import torch

from data.dataset import MultiDomainDataLoader, TimeSeriesDataset


class TestTimeSeriesDataset:
    """Tests for the TimeSeriesDataset class."""

    def test_univariate_sample_count(self) -> None:
        """Dataset with 1000 univariate steps yields correct number of samples."""
        data = np.random.randn(1000).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        # Valid starts: 1000 - (512 + 96) + 1 = 393
        expected = 1000 - 512 - 96 + 1
        assert len(ds) == expected

    def test_multivariate_sample_count(self) -> None:
        """Dataset with multivariate data yields samples for each channel."""
        data = np.random.randn(1000, 3).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        # 393 samples per channel × 3 channels
        expected_per_channel = 1000 - 512 - 96 + 1
        assert len(ds) == expected_per_channel * 3

    def test_output_shapes(self) -> None:
        """Each sample has correct context and target tensor shapes."""
        data = np.random.randn(700).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        ctx, tgt = ds[0]
        assert ctx.shape == (512,)
        assert tgt.shape == (96,)

    def test_output_dtype(self) -> None:
        """Output tensors are float32."""
        data = np.random.randn(700).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        ctx, tgt = ds[0]
        assert ctx.dtype == torch.float32
        assert tgt.dtype == torch.float32

    def test_context_target_contiguous(self) -> None:
        """Context window and target are contiguous in the original series."""
        # Use a simple sequential series so we can verify values
        data = np.arange(700, dtype=np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        ctx, tgt = ds[0]
        # First sample: context is [0, 511], target is [512, 607]
        assert ctx[0].item() == 0.0
        assert ctx[-1].item() == 511.0
        assert tgt[0].item() == 512.0
        assert tgt[-1].item() == 607.0

    def test_sliding_window_offset(self) -> None:
        """Second sample is offset by 1 from the first (stride=1 sliding window)."""
        data = np.arange(700, dtype=np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        ctx0, _ = ds[0]
        ctx1, _ = ds[1]
        # Second sample starts at index 1
        assert ctx1[0].item() == 1.0

    def test_short_series_empty_dataset(self) -> None:
        """Series shorter than context + horizon yields zero samples."""
        data = np.random.randn(500).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        assert len(ds) == 0

    def test_exact_minimum_length(self) -> None:
        """Series of exactly context + horizon length yields exactly 1 sample."""
        length = 512 + 96  # 608
        data = np.random.randn(length).astype(np.float32)
        ds = TimeSeriesDataset(data, context_length=512, forecast_horizon=96)
        assert len(ds) == 1


class TestMultiDomainDataLoader:
    """Tests for the MultiDomainDataLoader class."""

    def _make_dataset(self, length: int) -> TimeSeriesDataset:
        """Helper to create a dataset with given series length."""
        data = np.random.randn(length).astype(np.float32)
        return TimeSeriesDataset(data, context_length=512, forecast_horizon=96)

    def test_round_robin_order(self) -> None:
        """Batches are yielded in round-robin domain order."""
        ds1 = self._make_dataset(800)
        ds2 = self._make_dataset(900)
        ds3 = self._make_dataset(1000)
        loader = MultiDomainDataLoader(
            [ds1, ds2, ds3],
            batch_size=16,
            domain_names=["a", "b", "c"],
        )
        domains = [domain for _, _, domain in loader]
        # Check round-robin pattern
        for i, domain in enumerate(domains):
            expected = ["a", "b", "c"][i % 3]
            assert domain == expected, f"Batch {i}: expected {expected}, got {domain}"

    def test_epoch_stops_at_smallest_dataset(self) -> None:
        """Iteration stops when the smallest dataset is exhausted."""
        ds_small = self._make_dataset(650)  # Few samples
        ds_large = self._make_dataset(2000)  # Many samples
        loader = MultiDomainDataLoader(
            [ds_small, ds_large],
            batch_size=16,
            domain_names=["small", "large"],
        )
        # Count total batches
        total = sum(1 for _ in loader)
        # Should be limited by the smaller dataset
        from math import ceil
        min_batches = ceil(len(ds_small) / 16)
        expected_total = min_batches * 2
        assert total == expected_total

    def test_len_matches_iteration(self) -> None:
        """__len__ returns the same count as actual iteration."""
        ds1 = self._make_dataset(800)
        ds2 = self._make_dataset(1000)
        ds3 = self._make_dataset(900)
        loader = MultiDomainDataLoader(
            [ds1, ds2, ds3],
            batch_size=32,
            domain_names=["e", "w", "f"],
        )
        actual_count = sum(1 for _ in loader)
        assert len(loader) == actual_count

    def test_batch_shapes(self) -> None:
        """Batches have correct tensor shapes."""
        ds1 = self._make_dataset(800)
        ds2 = self._make_dataset(800)
        loader = MultiDomainDataLoader(
            [ds1, ds2],
            batch_size=8,
            domain_names=["a", "b"],
        )
        ctx, tgt, _ = next(iter(loader))
        assert ctx.shape[1] == 512
        assert tgt.shape[1] == 96
        assert ctx.shape[0] <= 8

    def test_num_domains_property(self) -> None:
        """num_domains property returns correct count."""
        ds1 = self._make_dataset(800)
        ds2 = self._make_dataset(800)
        ds3 = self._make_dataset(800)
        loader = MultiDomainDataLoader([ds1, ds2, ds3], batch_size=16)
        assert loader.num_domains == 3

    def test_default_domain_names(self) -> None:
        """Default domain names are generated when not provided."""
        ds1 = self._make_dataset(800)
        ds2 = self._make_dataset(800)
        loader = MultiDomainDataLoader([ds1, ds2], batch_size=16)
        assert loader.domain_names == ["domain_0", "domain_1"]
