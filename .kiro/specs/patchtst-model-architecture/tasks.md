# Implementation Plan: PatchTST Model Architecture

## Overview

This plan implements the complete PatchTST model architecture from scratch across five modules. The implementation follows a bottom-up dependency order: foundational components (patching, positional encoding) first, then the transformer encoder, then the probabilistic head, and finally the top-level assembly. Property-based tests are written alongside each module to validate correctness properties.

## Tasks

- [x] 1. Implement PatchEmbedding in model/patching.py
  - [x] 1.1 Create `model/patching.py` with ASCII-art documentation at the top showing how a raw time series of length 512 gets split into 63 overlapping patches (patch_len=16, stride=8), explaining that patches become "tokens" for the transformer
  - [x] 1.2 Implement `PatchEmbedding` class with `__init__` accepting `patch_len` (default 16), `d_model` (default 256), and `stride` (default 8), containing an `nn.Linear(patch_len, d_model)` projection layer
  - [x] 1.3 Implement `forward()` method that uses `torch.Tensor.unfold()` to extract patches from input shape (batch, seq_len), applies linear projection, and returns shape (batch, num_patches, d_model)
  - [x] 1.4 Add input validation in `forward()` raising `ValueError` if seq_len < patch_len
  - [x] 1.5 Write property test: for random valid lengths L >= 16, verify patch count equals `floor((L - 16) / 8) + 1`
  - [x] 1.6 Write property test: for random inputs (batch, seq_len), verify output shape is (batch, expected_num_patches, d_model)
  - [x] 1.7 Write property test: for overlapping patches, verify overlap region of patch[i] (last 8 values) matches beginning of patch[i+1] (first 8 values)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement SinusoidalPositionalEncoding in model/positional_encoding.py
  - [x] 2.1 Create `model/positional_encoding.py` with comment block showing the sin/cos formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)), and an ASCII visualization of the encoding pattern
  - [x] 2.2 Implement `SinusoidalPositionalEncoding` class with `__init__` accepting `d_model` (default 256), `max_len` (default 128), and `dropout` (default 0.1)
  - [x] 2.3 Pre-compute the sinusoidal encoding matrix and register it as a non-trainable buffer using `self.register_buffer()`
  - [x] 2.4 Add `nn.Dropout(dropout)` layer applied after adding positional encoding
  - [x] 2.5 Implement `forward()` method that adds positional encoding (sliced to seq_len) to input and applies dropout, preserving shape (batch, seq_len, d_model)
  - [x] 2.6 Write property test: for any two distinct positions, verify encoding vectors are non-identical (L2 distance > 0)
  - [x] 2.7 Write property test: for random input tensors, verify output shape equals input shape
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Implement TransformerEncoderBlock and TransformerEncoder in model/transformer_encoder.py
  - [x] 3.1 Create `model/transformer_encoder.py` with explanatory comments for each component: self-attention for cross-patch pattern detection (seasonality), LayerNorm for training stability, FFN for independent patch processing, residual connections for gradient flow
  - [x] 3.2 Implement `MultiHeadSelfAttention` class with Q/K/V linear projections, scaled dot-product attention (softmax(QK^T / sqrt(d_k)) * V), output projection, and attention dropout (d_model=256, n_heads=8, d_k=32)
  - [x] 3.3 Add validation in `MultiHeadSelfAttention.__init__` raising `ValueError` if d_model is not divisible by n_heads
  - [x] 3.4 Implement `TransformerEncoderBlock` class with pre-norm architecture: LayerNorm → MHSA → Dropout → Residual → LayerNorm → FFN (Linear 256→1024, GELU, Linear 1024→256) → Dropout → Residual
  - [x] 3.5 Implement `TransformerEncoder` class that stacks N_LAYERS=6 `TransformerEncoderBlock` instances in `nn.ModuleList` with a final `LayerNorm` after the last block
  - [x] 3.6 Implement `TransformerEncoder.forward()` that passes input sequentially through all layers then applies final LayerNorm
  - [x] 3.7 Write property test: for random inputs (batch, seq_len, d_model), verify output shape matches input shape through the full encoder
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Implement ProbabilisticHead and quantile_loss in model/probabilistic_head.py
  - [x] 4.1 Create `model/probabilistic_head.py` with explanatory comments about probabilistic forecasting: P10 as lower bound (10% chance actual is below), P50 as median prediction, P90 as upper bound (90% chance actual is below)
  - [x] 4.2 Implement `ProbabilisticHead` class with three separate `nn.Linear` heads (`head_p10`, `head_p50`, `head_p90`), each mapping `(num_patches * d_model)` → `forecast_horizon`
  - [x] 4.3 Implement `ProbabilisticHead.forward()` that flattens encoder output, passes through 3 heads, stacks results into (batch, forecast_horizon, 3), and applies `torch.sort` for monotonicity enforcement (P10 <= P50 <= P90)
  - [x] 4.4 Add input validation raising `ValueError` if `encoder_output.shape[1] != num_patches`
  - [x] 4.5 Implement `quantile_loss()` function with pinball loss formula: `tau * max(y - q_hat, 0) + (1 - tau) * max(q_hat - y, 0)`, returning mean loss across all dimensions
  - [x] 4.6 Write property test: for random encoder outputs, verify output satisfies P10 <= P50 <= P90 at every position
  - [x] 4.7 Write property test: for random encoder outputs, verify output shape is (batch, forecast_horizon, 3)
  - [x] 4.8 Write property test: for random predictions and targets, verify quantile_loss returns non-negative scalar
  - [x] 4.9 Write property test: when predictions equal targets, verify quantile_loss returns zero
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 5. Implement PatchTST assembly in model/patch_tst.py
  - [x] 5.1 Create `model/patch_tst.py` with the full `PatchTST` class composing `PatchEmbedding`, `SinusoidalPositionalEncoding`, `TransformerEncoder`, and `ProbabilisticHead`
  - [x] 5.2 Implement `__init__` that instantiates all submodules from config, calls `count_parameters()`, prints the count, and prints a warning if outside 8-12M range
  - [x] 5.3 Implement `forward()` method with input/output shape comments: input (batch, seq_len) → patch_embed → pos_enc → encoder → head → output (batch, forecast_horizon, 3)
  - [x] 5.4 Implement `forecast()` method that calls `forward()`, splits output into P10/P50/P90 channels, denormalizes using provided mean/std, and returns dict with keys 'p10', 'p50', 'p90' each of shape (batch, forecast_horizon)
  - [x] 5.5 Implement `count_parameters()` method returning sum of all trainable parameters
  - [x] 5.6 Write property test: for input (batch, 512), verify output shape is (batch, 96, 3)
  - [x] 5.7 Write property test: verify `forecast()` returns dict with keys 'p10', 'p50', 'p90' and correct tensor shapes (batch, forecast_horizon)
  - [x] 5.8 Write unit test: verify parameter count is between 8M and 12M with default config
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3_

## Notes

- All modules import configuration from `config.py` for default hyperparameters
- Property-based tests use Hypothesis with PyTorch tensor strategies
- Existing files (`model/patchtst.py`, `model/attention.py`, `model/encoder.py`, `model/patch_embedding.py`, `model/transformer_layer.py`, `forecasting/probabilistic_head.py`) are preserved for backward compatibility
- The new modules use distinct filenames (`model/patching.py`, `model/positional_encoding.py`, `model/transformer_encoder.py`, `model/probabilistic_head.py`, `model/patch_tst.py`) to avoid conflicts
- Note: `model/probabilistic_head.py` (new) is distinct from `forecasting/probabilistic_head.py` (existing)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3", "2.4", "2.5"] },
    { "id": 1, "tasks": ["1.5", "1.6", "1.7", "2.6", "2.7", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6"] },
    { "id": 2, "tasks": ["3.7", "4.1", "4.2", "4.3", "4.4", "4.5"] },
    { "id": 3, "tasks": ["4.6", "4.7", "4.8", "4.9", "5.1", "5.2", "5.3", "5.4", "5.5"] },
    { "id": 4, "tasks": ["5.6", "5.7", "5.8"] }
  ]
}
```
