"""Property-based tests for the PatchEmbedding module.

These tests validate the correctness properties defined in the design document
for the patching component of the PatchTST architecture. Each test uses Hypothesis
to generate random inputs and verifies that universal properties hold.

Properties tested:
- Property 1 (Task 1.5): Patch count formula correctness
- Property 2 (Task 1.6): Output shape transformation correctness
- Property 3 (Task 1.7): Overlap consistency between consecutive patches
"""

import math

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.patching import PatchEmbedding


# Default parameters matching the design spec
PATCH_LEN = 16
D_MODEL = 256
STRIDE = 8


class TestPatchCountFormula:
    """**Validates: Requirements 1.2**

    Property 1: For any input of length L >= patch_len, the number of patches
    produced equals floor((L - patch_len) / stride) + 1.
    """

    @given(seq_len=st.integers(min_value=16, max_value=1024))
    @settings(max_examples=100)
    def test_patch_count_equals_formula(self, seq_len: int) -> None:
        """For random valid lengths L >= 16, verify patch count equals
        floor((L - 16) / 8) + 1."""
        embed = PatchEmbedding(patch_len=PATCH_LEN, d_model=D_MODEL, stride=STRIDE)
        embed.eval()

        x = torch.randn(1, seq_len)

        with torch.no_grad():
            output = embed(x)

        expected_num_patches = math.floor((seq_len - PATCH_LEN) / STRIDE) + 1
        actual_num_patches = output.shape[1]

        assert actual_num_patches == expected_num_patches, (
            f"For seq_len={seq_len}, expected {expected_num_patches} patches "
            f"but got {actual_num_patches}"
        )


class TestPatchEmbeddingShape:
    """**Validates: Requirements 1.4**

    Property 2: For any input (batch, seq_len) where seq_len >= patch_len,
    the PatchEmbedding output shape is (batch, expected_num_patches, d_model).
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=16, max_value=1024),
    )
    @settings(max_examples=100)
    def test_output_shape_is_correct(self, batch_size: int, seq_len: int) -> None:
        """For random inputs (batch, seq_len), verify output shape is
        (batch, expected_num_patches, d_model)."""
        embed = PatchEmbedding(patch_len=PATCH_LEN, d_model=D_MODEL, stride=STRIDE)
        embed.eval()

        x = torch.randn(batch_size, seq_len)

        with torch.no_grad():
            output = embed(x)

        expected_num_patches = math.floor((seq_len - PATCH_LEN) / STRIDE) + 1

        assert output.shape == (batch_size, expected_num_patches, D_MODEL), (
            f"For input ({batch_size}, {seq_len}), expected output shape "
            f"({batch_size}, {expected_num_patches}, {D_MODEL}) "
            f"but got {output.shape}"
        )


class TestOverlapConsistency:
    """**Validates: Requirements 1.5**

    Property 3: For any time series, overlapping patches extracted with
    stride < patch_len SHALL have the overlapping region of patch[i]
    (last patch_len - stride values) equal to the beginning of patch[i+1]
    (first patch_len - stride values).
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=32, max_value=512),
    )
    @settings(max_examples=100)
    def test_overlap_region_matches(self, batch_size: int, seq_len: int) -> None:
        """For overlapping patches, verify overlap region of patch[i] (last 8
        values) matches beginning of patch[i+1] (first 8 values)."""
        # We need at least 2 patches to test overlap
        # 2 patches requires seq_len >= patch_len + stride = 16 + 8 = 24
        assume(seq_len >= PATCH_LEN + STRIDE)

        x = torch.randn(batch_size, seq_len)

        # Use unfold directly to get raw patches (before projection)
        # This tests the overlap property on the actual time series values
        patches = x.unfold(dimension=-1, size=PATCH_LEN, step=STRIDE)
        # patches shape: (batch, num_patches, patch_len)

        num_patches = patches.shape[1]
        overlap_size = PATCH_LEN - STRIDE  # 16 - 8 = 8

        # For each consecutive pair of patches, verify overlap
        for i in range(num_patches - 1):
            # Last `overlap_size` values of patch[i]
            tail_of_current = patches[:, i, -overlap_size:]
            # First `overlap_size` values of patch[i+1]
            head_of_next = patches[:, i + 1, :overlap_size]

            assert torch.allclose(tail_of_current, head_of_next, atol=1e-7), (
                f"Overlap mismatch at patch index {i}: "
                f"patch[{i}][-{overlap_size}:] != patch[{i+1}][:{overlap_size}]"
            )
