"""Unit tests for the data/patching.py module.

Tests cover the core patching functions: compute_num_patches and create_patches.
Validates correct patch counts, shapes, content, trailing-step discarding, and
error handling for invalid inputs.

Related modules:
    - data/patching.py: The module under test
    - config.py: Provides default PATCH_LEN=16, PATCH_STRIDE=8
"""

import numpy as np
import pytest

from data.patching import compute_num_patches, create_patches


class TestComputeNumPatches:
    """Tests for the compute_num_patches function."""

    def test_standard_context_window(self) -> None:
        """512 time steps with patch_len=16, stride=8 yields 63 patches."""
        assert compute_num_patches(512, patch_len=16, stride=8) == 63

    def test_minimum_length_one_patch(self) -> None:
        """Series exactly equal to patch_len yields exactly 1 patch."""
        assert compute_num_patches(16, patch_len=16, stride=8) == 1

    def test_two_patches(self) -> None:
        """Series of length 24 with stride 8 yields 2 patches (starts at 0 and 8)."""
        assert compute_num_patches(24, patch_len=16, stride=8) == 2

    def test_trailing_steps_not_counted(self) -> None:
        """Extra steps beyond the last complete patch are not counted."""
        # 25 steps: patches at [0:16] and [8:24], step 24 is trailing
        assert compute_num_patches(25, patch_len=16, stride=8) == 2

    def test_non_overlapping_stride(self) -> None:
        """When stride equals patch_len, patches are non-overlapping."""
        # 64 steps / 16 stride = 4 patches
        assert compute_num_patches(64, patch_len=16, stride=16) == 4

    def test_stride_one(self) -> None:
        """Stride of 1 produces maximum number of patches."""
        # floor((20 - 16) / 1) + 1 = 5
        assert compute_num_patches(20, patch_len=16, stride=1) == 5

    def test_raises_on_short_series(self) -> None:
        """Series shorter than patch_len raises ValueError."""
        with pytest.raises(ValueError, match="must be >= patch_len"):
            compute_num_patches(15, patch_len=16, stride=8)

    def test_raises_on_zero_patch_len(self) -> None:
        """patch_len of 0 raises ValueError."""
        with pytest.raises(ValueError, match="patch_len must be positive"):
            compute_num_patches(100, patch_len=0, stride=8)

    def test_raises_on_negative_stride(self) -> None:
        """Negative stride raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            compute_num_patches(100, patch_len=16, stride=-1)


class TestCreatePatches:
    """Tests for the create_patches function."""

    def test_output_shape_standard(self) -> None:
        """512-step series produces (63, 16) patch array."""
        series = np.random.randn(512).astype(np.float32)
        patches = create_patches(series, patch_len=16, stride=8)
        assert patches.shape == (63, 16)

    def test_output_shape_minimum(self) -> None:
        """16-step series produces exactly (1, 16) patch array."""
        series = np.arange(16, dtype=np.float64)
        patches = create_patches(series, patch_len=16, stride=8)
        assert patches.shape == (1, 16)

    def test_patch_content_correctness(self) -> None:
        """Each patch contains the correct slice of the original series."""
        series = np.arange(32, dtype=np.float64)
        patches = create_patches(series, patch_len=16, stride=8)
        # Patch 0: indices 0-15
        np.testing.assert_array_equal(patches[0], series[0:16])
        # Patch 1: indices 8-23
        np.testing.assert_array_equal(patches[1], series[8:24])
        # Patch 2: indices 16-31
        np.testing.assert_array_equal(patches[2], series[16:32])

    def test_trailing_steps_discarded(self) -> None:
        """Steps beyond the last complete patch are not included."""
        # 25 steps: only 2 patches (0:16, 8:24), step 24 is discarded
        series = np.arange(25, dtype=np.float64)
        patches = create_patches(series, patch_len=16, stride=8)
        assert patches.shape == (2, 16)
        # Verify last patch ends at index 23, not 24
        np.testing.assert_array_equal(patches[1], series[8:24])

    def test_preserves_dtype(self) -> None:
        """Output dtype matches input dtype."""
        series_f32 = np.ones(32, dtype=np.float32)
        patches_f32 = create_patches(series_f32, patch_len=16, stride=8)
        assert patches_f32.dtype == np.float32

        series_f64 = np.ones(32, dtype=np.float64)
        patches_f64 = create_patches(series_f64, patch_len=16, stride=8)
        assert patches_f64.dtype == np.float64

    def test_raises_on_2d_input(self) -> None:
        """2-D input raises ValueError (channel-independent design)."""
        with pytest.raises(ValueError, match="1-D series"):
            create_patches(np.zeros((32, 2)), patch_len=16, stride=8)

    def test_raises_on_short_series(self) -> None:
        """Series shorter than patch_len raises ValueError."""
        with pytest.raises(ValueError, match="must be >= patch_len"):
            create_patches(np.zeros(10), patch_len=16, stride=8)

    def test_non_overlapping_patches(self) -> None:
        """When stride == patch_len, patches tile the series without overlap."""
        series = np.arange(48, dtype=np.float64)
        patches = create_patches(series, patch_len=16, stride=16)
        assert patches.shape == (3, 16)
        # Patches should be non-overlapping consecutive slices
        np.testing.assert_array_equal(patches[0], series[0:16])
        np.testing.assert_array_equal(patches[1], series[16:32])
        np.testing.assert_array_equal(patches[2], series[32:48])
