# Design Document: PatchTST Model Architecture

## Overview

This design specifies the complete PatchTST model architecture built from scratch across five modules. The architecture follows the PatchTST paper's channel-independent design: raw univariate time series → overlapping patches → linear embedding → sinusoidal positional encoding → 6 transformer encoder blocks → probabilistic forecasting head producing P10/P50/P90 quantile predictions.

The implementation refactors the existing model files to add sinusoidal positional encoding (replacing learnable), restructures the probabilistic head with separate linear heads per quantile, and creates a unified assembly module with both `forward()` and `forecast()` methods.

## Architecture

```
Input: (batch, 512)
    │
    ▼
┌─────────────────────────────────┐
│  model/patching.py              │
│  PatchEmbedding                 │
│  - Unfold: (B,512) → (B,63,16) │
│  - Linear: (B,63,16) → (B,63,256)│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  model/positional_encoding.py   │
│  SinusoidalPositionalEncoding   │
│  - Add sin/cos encoding         │
│  - (B,63,256) → (B,63,256)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  model/transformer_encoder.py   │
│  6x TransformerEncoderBlock     │
│  - MultiHeadSelfAttention (8h)  │
│  - LayerNorm (pre-norm)         │
│  - FFN: 256→1024→256 (GELU)    │
│  - Residual connections         │
│  - Final LayerNorm              │
│  - (B,63,256) → (B,63,256)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  model/probabilistic_head.py    │
│  ProbabilisticHead              │
│  - 3 separate Linear heads      │
│  - P10, P50, P90 quantiles      │
│  - Monotonicity enforcement     │
│  - (B,63,256) → (B,96,3)       │
└─────────────────────────────────┘
    │
    ▼
Output: (batch, 96, 3) = [P10, P50, P90]
```

## Components and Interfaces

### Component 1: model/patching.py — PatchEmbedding

**Purpose:** Segment raw time series into overlapping patches and project each patch into embedding space.

**Design Decisions:**
- Use `torch.Tensor.unfold()` for efficient patch extraction (no explicit loops)
- Linear projection (nn.Linear) maps patch_len=16 → d_model=256, identical to ViT's patch projection
- Include ASCII-art at module top showing the patching process visually
- The class handles both patching and embedding in one step for clean API

**Interface:**
```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int = 16, d_model: int = 256, stride: int = 8):
        # nn.Linear(patch_len, d_model) for projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) → patches: (batch, num_patches, patch_len) → embedded: (batch, num_patches, d_model)
```

**ASCII Documentation (included at top of file):**
```
# Raw time series (length 512):
# [──────────────────────────────────────────────────────────────────]
#  t=0                                                            t=511
#
# Patch extraction (patch_len=16, stride=8):
# Patch 0:  [████████████████]                    (t=0  to t=15)
# Patch 1:          [████████████████]            (t=8  to t=23)
# Patch 2:                  [████████████████]    (t=16 to t=31)
# ...
# Patch 62:                              [████████████████]  (t=496 to t=511)
#
# Result: 63 patches, each a "token" for the transformer
# Formula: num_patches = floor((512 - 16) / 8) + 1 = 63
```

### Component 2: model/positional_encoding.py — SinusoidalPositionalEncoding

**Purpose:** Add fixed sinusoidal position information to patch embeddings so the transformer can distinguish patch order.

**Design Decisions:**
- Use fixed (non-learnable) sinusoidal encoding from "Attention is All You Need"
- Register as a buffer (not a parameter) — no gradient computation, persists on device transfers
- Pre-compute the full encoding matrix at init for max_len positions
- Slice to actual sequence length in forward() for flexibility
- Apply dropout after adding positional encoding

**Interface:**
```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 128, dropout: float = 0.1):
        # Pre-compute sin/cos encoding matrix, register as buffer
        # Apply dropout after adding positional encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) → x + PE[:seq_len]: (batch, seq_len, d_model)
```

**Encoding Formula (in comments):**
```
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# Visualization (positions 0-7, dimensions 0-15):
#   dim→  0    1    2    3    4    5    6    7   ...
# pos=0  sin  cos  sin  cos  sin  cos  sin  cos
# pos=1  ░░░  ░░░  ░▒▒  ▒▒▒  ▒▒▓  ▓▓▓  ▓▓█  ███  (high freq → low freq)
# pos=2  ▒▒▒  ▒▒▒  ▒▓▓  ▓▓▓  ▓▓█  ███  ███  ███
# ...
# Lower dimensions oscillate rapidly (local patterns)
# Higher dimensions oscillate slowly (global patterns)
```

### Component 3: model/transformer_encoder.py — TransformerEncoderBlock + TransformerEncoder

**Purpose:** Process patch embeddings through self-attention and feedforward layers to learn contextual relationships between patches.

**Design Decisions:**
- Pre-norm architecture (LayerNorm before sublayer) for training stability
- Multi-head self-attention with 8 heads (d_k = 256/8 = 32 per head)
- FFN with GELU activation (smoother than ReLU, standard in modern transformers)
- Stack 6 blocks with a final LayerNorm after the last block
- Residual connections around both sublayers
- Detailed comments explaining each component's role in time series context

**Interface:**
```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        # norm1, attention, norm2, ffn, dropout layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm → MHSA → residual → Pre-norm → FFN → residual
        # (batch, num_patches, d_model) → (batch, num_patches, d_model)

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int = 6, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        # ModuleList of n_layers TransformerEncoderBlocks + final LayerNorm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential pass through all layers + final norm
```

### Component 4: model/probabilistic_head.py — ProbabilisticHead + quantile_loss

**Purpose:** Map encoder output to probabilistic forecasts with three quantile levels and provide the training loss function.

**Design Decisions:**
- Three separate nn.Linear heads (one per quantile) rather than a single head with 3 outputs — allows each head to specialize
- Flatten encoder output → project to forecast_horizon per head → stack and sort for monotonicity
- Quantile loss (pinball loss) implemented as a standalone function for reuse
- Monotonicity enforced via torch.sort along the quantile dimension

**Interface:**
```python
class ProbabilisticHead(nn.Module):
    def __init__(self, d_model: int = 256, num_patches: int = 63, forecast_horizon: int = 96, quantiles: list = [0.1, 0.5, 0.9]):
        # self.head_p10 = nn.Linear(num_patches * d_model, forecast_horizon)
        # self.head_p50 = nn.Linear(num_patches * d_model, forecast_horizon)
        # self.head_p90 = nn.Linear(num_patches * d_model, forecast_horizon)
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output: (batch, num_patches, d_model)
        # → flatten: (batch, num_patches * d_model)
        # → 3 heads: each (batch, forecast_horizon)
        # → stack + sort: (batch, forecast_horizon, 3)

def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]) -> torch.Tensor:
    # Pinball loss: tau * max(y - q, 0) + (1-tau) * max(q - y, 0)
    # Returns scalar mean loss
```

### Component 5: model/patch_tst.py — PatchTST Assembly

**Purpose:** Compose all components into the complete end-to-end model with training and inference interfaces.

**Design Decisions:**
- Single class `PatchTST` that owns all submodules
- `forward()` for training: raw series → quantile forecasts (normalized space)
- `forecast()` for inference: raw series + normalization stats → denormalized P10/P50/P90
- Print parameter count at initialization with warning if outside 8-12M range
- Accept mean/std in forecast() for denormalization

**Interface:**
```python
class PatchTST(nn.Module):
    def __init__(self, config=Config):
        # self.patch_embedding = PatchEmbedding(...)
        # self.positional_encoding = SinusoidalPositionalEncoding(...)
        # self.encoder = TransformerEncoder(...)
        # self.head = ProbabilisticHead(...)
        # Print parameter count at init
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) → (batch, forecast_horizon, 3)
        # Pipeline: patch_embed → pos_enc → encoder → head
    
    def forecast(self, x: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> dict:
        # Returns {'p10': tensor, 'p50': tensor, 'p90': tensor}
        # Each tensor: (batch, forecast_horizon) — denormalized
    
    def count_parameters(self) -> int:
        # Sum of all trainable parameters
```

## Data Models

### Tensor Shapes Through the Pipeline

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | `(batch, 512)` | Raw univariate time series |
| After unfold | `(batch, 63, 16)` | Overlapping patches |
| After embedding | `(batch, 63, 256)` | Projected patch embeddings |
| After positional encoding | `(batch, 63, 256)` | Position-aware embeddings |
| After encoder | `(batch, 63, 256)` | Contextualized representations |
| After head | `(batch, 96, 3)` | Quantile forecasts [P10, P50, P90] |

### Configuration Parameters

| Parameter | Value | Used By |
|-----------|-------|---------|
| `D_MODEL` | 256 | All modules |
| `N_HEADS` | 8 | Transformer encoder |
| `N_LAYERS` | 6 | Transformer encoder |
| `D_FF` | 1024 | Transformer encoder |
| `DROPOUT` | 0.1 | All modules |
| `PATCH_LEN` | 16 | Patching |
| `PATCH_STRIDE` | 8 | Patching |
| `CONTEXT_LENGTH` | 512 | Assembly |
| `NUM_PATCHES` | 63 | All modules |
| `FORECAST_HORIZON` | 96 | Probabilistic head |
| `QUANTILES` | [0.1, 0.5, 0.9] | Probabilistic head |

## Error Handling

| Error Condition | Module | Behavior |
|----------------|--------|----------|
| Input seq_len < patch_len | PatchEmbedding | Raise ValueError with descriptive message |
| d_model not divisible by n_heads | TransformerEncoderBlock | Raise ValueError at init |
| Encoder output patches != expected num_patches | ProbabilisticHead | Raise ValueError with shape info |
| Parameter count outside 8-12M | PatchTST assembly | Print warning (non-fatal) |

## Testing Strategy

- **Property-based tests** for shape invariants, monotonicity, loss non-negativity, and overlap consistency
- **Unit tests** for each module in isolation with known input/output pairs
- **Integration test** for the full pipeline end-to-end shape verification
- **Parameter count test** to verify the model stays within 8-12M budget

## Correctness Properties

### Property 1: Patch Count Formula

*For any* input of length L >= patch_len, the number of patches produced equals floor((L - patch_len) / stride) + 1. Generate random valid lengths, verify patch count matches formula.

**Validates: Requirements 1.2**

### Property 2: Patch Embedding Shape Transformation

*For any* input (batch, seq_len) where seq_len >= patch_len, the PatchEmbedding output shape is (batch, expected_num_patches, d_model). Generate random batch sizes and valid sequence lengths, verify output dimensions.

**Validates: Requirements 1.4**

### Property 3: Overlap Consistency

*For any* time series, overlapping patches extracted with stride < patch_len SHALL have the overlapping region of patch[i] (last patch_len - stride values) equal to the beginning of patch[i+1] (first patch_len - stride values).

**Validates: Requirements 1.5**

### Property 4: Positional Encoding Uniqueness

*For any* two distinct position indices pos1 and pos2, the sinusoidal encoding vectors SHALL be non-identical (L2 distance > 0).

**Validates: Requirements 2.5**

### Property 5: Positional Encoding Shape Preservation

*For any* input tensor of shape (batch, seq_len, d_model), adding sinusoidal positional encoding SHALL preserve the tensor shape exactly.

**Validates: Requirements 2.3**

### Property 6: Transformer Encoder Shape Invariant

*For any* input tensor of shape (batch, seq_len, d_model), the transformer encoder SHALL produce output of the same shape through all 6 layers.

**Validates: Requirements 3.3, 3.5**

### Property 7: Probabilistic Head Monotonicity

*For any* encoder output tensor, the ProbabilisticHead output SHALL satisfy P10 <= P50 <= P90 at every forecast timestep and batch element.

**Validates: Requirements 4.5**

### Property 8: Probabilistic Head Output Shape

*For any* encoder output of shape (batch, num_patches, d_model), the ProbabilisticHead SHALL produce output of shape (batch, forecast_horizon, 3).

**Validates: Requirements 4.2**

### Property 9: Quantile Loss Non-Negativity

*For any* prediction and target tensor combination, the quantile_loss function SHALL return a non-negative scalar value.

**Validates: Requirements 6.5**

### Property 10: Quantile Loss Zero at Perfect Prediction

*For any* target tensor, when predictions exactly equal targets for all quantiles, the quantile_loss SHALL return zero.

**Validates: Requirements 6.2**

### Property 11: Full Model End-to-End Shape

*For any* input of shape (batch, 512), the full PatchTST model SHALL produce output of shape (batch, 96, 3).

**Validates: Requirements 5.2**

### Property 12: Forecast Method Returns Correct Structure

*For any* valid input, the forecast() method SHALL return a dictionary with keys 'p10', 'p50', 'p90', each containing a tensor of shape (batch, forecast_horizon).

**Validates: Requirements 5.3, 5.6**

## Dependencies

- **PyTorch** (torch, torch.nn, torch.nn.functional): Core deep learning framework
- **math**: For sqrt in attention scaling
- **config.py**: Central configuration (D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, PATCH_LEN, PATCH_STRIDE, CONTEXT_LENGTH, NUM_PATCHES, FORECAST_HORIZON, QUANTILES)

## Migration Notes

The existing codebase already has implementations in `model/patchtst.py`, `model/attention.py`, `model/encoder.py`, `model/patch_embedding.py`, `model/transformer_layer.py`, `data/patching.py`, and `forecasting/probabilistic_head.py`. The new implementation:

1. **model/patching.py** — New file. Replaces the patching logic from `data/patching.py` and embedding from `model/patch_embedding.py` into a single module with ASCII documentation.
2. **model/positional_encoding.py** — New file. Replaces the learnable positional encoding in `model/patch_embedding.py` with sinusoidal encoding.
3. **model/transformer_encoder.py** — New file. Consolidates `model/attention.py`, `model/transformer_layer.py`, and `model/encoder.py` into one documented module.
4. **model/probabilistic_head.py** — New file. Replaces `forecasting/probabilistic_head.py` with separate heads per quantile.
5. **model/patch_tst.py** — New file. Replaces `model/patchtst.py` with the full assembly including forecast() method.

Existing files are preserved for backward compatibility until migration is complete.
