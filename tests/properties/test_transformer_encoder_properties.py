"""Property-based tests for TransformerEncoder.

Tests verify the shape invariant property of the transformer encoder:
- For any input (batch, seq_len, d_model), the output shape matches the input shape
  through the full 6-layer encoder stack.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from model.transformer_encoder import TransformerEncoder

# Module defaults matching config
N_LAYERS = 6
D_MODEL = 256
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.1


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=63),
)
@settings(max_examples=200)
def test_encoder_output_shape_matches_input_shape(batch_size: int, seq_len: int):
    """For random inputs (batch, seq_len, d_model), output shape matches input shape through the full encoder.

    **Validates: Requirements 3.3, 3.5**
    """
    encoder = TransformerEncoder(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )
    encoder.eval()

    x = torch.randn(batch_size, seq_len, D_MODEL)

    with torch.no_grad():
        output = encoder(x)

    assert output.shape == x.shape, (
        f"Output shape {output.shape} does not match input shape {x.shape}. "
        f"Expected ({batch_size}, {seq_len}, {D_MODEL})."
    )
