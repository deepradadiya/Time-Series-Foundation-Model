"""Property-based tests for SinusoidalPositionalEncoding.

Tests verify correctness properties of the positional encoding module:
- Uniqueness: distinct positions produce distinct encoding vectors
- Shape preservation: output shape matches input shape
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from model.positional_encoding import SinusoidalPositionalEncoding

# Module defaults
D_MODEL = 256
MAX_LEN = 128
DROPOUT = 0.1


@given(
    pos1=st.integers(min_value=0, max_value=MAX_LEN - 1),
    pos2=st.integers(min_value=0, max_value=MAX_LEN - 1),
)
@settings(max_examples=200)
def test_distinct_positions_have_nonidentical_encodings(pos1: int, pos2: int):
    """For any two distinct positions, encoding vectors are non-identical (L2 distance > 0).

    **Validates: Requirements 2.5**
    """
    # Only test distinct positions
    if pos1 == pos2:
        return

    encoder = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=MAX_LEN, dropout=DROPOUT)

    # Extract encoding vectors directly from the pe buffer
    # pe shape: (1, max_len, d_model)
    pe_buffer = encoder.pe

    vec1 = pe_buffer[0, pos1, :]
    vec2 = pe_buffer[0, pos2, :]

    # L2 distance must be strictly positive for distinct positions
    l2_distance = torch.norm(vec1 - vec2, p=2).item()
    assert l2_distance > 0, (
        f"Encoding vectors for positions {pos1} and {pos2} are identical "
        f"(L2 distance = {l2_distance})"
    )


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=MAX_LEN),
)
@settings(max_examples=200)
def test_output_shape_equals_input_shape(batch_size: int, seq_len: int):
    """For random input tensors, output shape equals input shape.

    **Validates: Requirements 2.3**
    """
    encoder = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=MAX_LEN, dropout=DROPOUT)
    # Use eval mode to disable dropout randomness for deterministic shape check
    encoder.eval()

    x = torch.randn(batch_size, seq_len, D_MODEL)
    output = encoder(x)

    assert output.shape == x.shape, (
        f"Output shape {output.shape} does not match input shape {x.shape}"
    )
