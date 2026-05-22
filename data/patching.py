"""Patch creation logic for the Time Series Foundation Model.

This module segments raw (normalized) time series into fixed-length overlapping
patches that serve as the atomic input tokens for the PatchTST transformer.
Patching reduces the effective sequence length (e.g., 512 → 63 tokens) while
preserving local temporal structure within each patch.

Related modules:
    - config.py: Provides PATCH_LEN (16), PATCH_STRIDE (8), and CONTEXT_LENGTH (512)
    - data/preprocess.py: Normalizes data before patching
    - data/dataset.py: Calls create_patches to build training samples
    - model/patch_embedding.py: Projects each patch into the transformer embedding space
"""

import numpy as np


def compute_num_patches(series_length: int, patch_len: int = 16, stride: int = 8) -> int:
    """Calculate the number of patches produced from a series of given length.

    Uses the formula: floor((L - patch_len) / stride) + 1
    This counts how many complete, non-overlapping-start patches fit within the
    series. Any trailing time steps that cannot form a full patch are discarded.

    Args:
        series_length: Total number of time steps in the input series (L).
        patch_len: Number of time steps in each patch. Defaults to 16.
        stride: Step size between the start of consecutive patches. Defaults to 8.

    Returns:
        The number of complete patches that can be extracted from the series.

    Raises:
        ValueError: If series_length is less than patch_len (cannot form even
            one complete patch), or if patch_len or stride are not positive.

    Examples:
        >>> compute_num_patches(512, patch_len=16, stride=8)
        63
        >>> compute_num_patches(16, patch_len=16, stride=8)
        1
        >>> compute_num_patches(24, patch_len=16, stride=8)
        2
    """
    # Validate inputs — patch_len and stride must be positive integers
    if patch_len <= 0:
        raise ValueError(f"patch_len must be positive, got {patch_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    # Series must be at least as long as one patch to produce any output
    if series_length < patch_len:
        raise ValueError(
            f"series_length ({series_length}) must be >= patch_len ({patch_len}) "
            f"to form at least one complete patch"
        )

    # Core formula: how many patches fit with the given stride
    # floor((L - patch_len) / stride) + 1
    num_patches = (series_length - patch_len) // stride + 1

    return num_patches


def create_patches(series: np.ndarray, patch_len: int = 16, stride: int = 8) -> np.ndarray:
    """Segment a 1-D time series into overlapping patches of fixed length.

    Extracts consecutive windows of size `patch_len` starting every `stride`
    steps. Trailing time steps that do not form a complete patch are discarded.
    For example, with the default settings (patch_len=16, stride=8) and a
    context window of 512 steps, this produces 63 patches with 50% overlap.

    Args:
        series: A 1-D numpy array of shape (L,) representing a univariate
            time series (typically already normalized).
        patch_len: Number of time steps per patch. Defaults to 16.
        stride: Number of time steps between the start of consecutive patches.
            Defaults to 8. When stride < patch_len, patches overlap.

    Returns:
        A 2-D numpy array of shape (num_patches, patch_len) where num_patches
        is computed as floor((L - patch_len) / stride) + 1. Each row is one
        patch containing `patch_len` consecutive time steps from the series.

    Raises:
        ValueError: If the input series is not 1-D, or if it is shorter than
            patch_len (cannot form even one patch), or if patch_len/stride
            are not positive.

    Examples:
        >>> series = np.arange(24, dtype=np.float32)
        >>> patches = create_patches(series, patch_len=16, stride=8)
        >>> patches.shape
        (2, 16)
        >>> patches[0]  # First patch: steps 0-15
        array([ 0.,  1.,  2., ..., 15.], dtype=float32)
        >>> patches[1]  # Second patch: steps 8-23
        array([ 8.,  9., 10., ..., 23.], dtype=float32)
    """
    # Ensure the input is a 1-D array (single univariate channel)
    if series.ndim != 1:
        raise ValueError(
            f"Expected a 1-D series array, got shape {series.shape}. "
            f"Each channel should be processed independently."
        )

    # Compute how many complete patches we can extract
    num_patches = compute_num_patches(len(series), patch_len, stride)

    # Pre-allocate the output array for efficiency
    # Shape: (num_patches, patch_len) — each row is one patch
    patches = np.empty((num_patches, patch_len), dtype=series.dtype)

    # Extract each patch by slicing the series at the appropriate offset
    for i in range(num_patches):
        # Starting index of the i-th patch
        start = i * stride
        # Ending index (exclusive) — always start + patch_len
        end = start + patch_len
        # Copy the patch into the output array
        patches[i] = series[start:end]

    return patches
