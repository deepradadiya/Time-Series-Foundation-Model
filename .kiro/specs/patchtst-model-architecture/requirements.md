# Requirements Document

## Introduction

This specification defines the complete PatchTST model architecture built from scratch. The architecture transforms raw univariate time series into probabilistic forecasts (P10/P50/P90) through a pipeline of patching, embedding, positional encoding, transformer encoding, and a probabilistic output head. The implementation spans five modules: patching, patch embedding, positional encoding, transformer encoder, probabilistic head, and a top-level assembly module. Each module includes detailed ASCII-art explanations and educational comments describing the mathematical intuition.

## Glossary

- **Patch_Embedding_Module**: The module (`model/patching.py`) responsible for segmenting raw time series into overlapping patches and projecting each patch into a dense embedding vector via a linear layer.
- **Positional_Encoding_Module**: The module (`model/positional_encoding.py`) that implements sinusoidal positional encoding from the "Attention is All You Need" paper, adding order information to patch embeddings.
- **Transformer_Encoder_Module**: The module (`model/transformer_encoder.py`) containing the TransformerEncoderBlock class (multi-head self-attention, LayerNorm, feedforward with GELU, residual connections) and a stack of N_LAYERS=6 blocks.
- **Probabilistic_Head_Module**: The module (`model/probabilistic_head.py`) that maps encoder output to three quantile forecasts (P10, P50, P90) using separate linear heads and quantile (pinball) loss.
- **PatchTST_Assembly_Module**: The top-level module (`model/patch_tst.py`) that composes all components into the full PatchTST model with forward() and forecast() methods.
- **Patch**: A contiguous segment of the time series of length patch_len (16 time steps) extracted with a sliding window of stride 8.
- **Quantile_Loss**: The pinball loss function that asymmetrically penalizes under-predictions and over-predictions based on the quantile level.
- **Denormalization**: The process of reversing z-score normalization to convert model predictions back to the original scale of the time series.

## Requirements

### Requirement 1: Patch Creation with ASCII Documentation

**User Story:** As a developer, I want the patching module to include ASCII-art documentation showing how a raw time series of length 512 is split into overlapping patches, so that the patching logic is immediately understandable.

#### Acceptance Criteria

1. THE Patch_Embedding_Module SHALL include an ASCII-art diagram at the top of the file showing a raw time series of length 512 being segmented into overlapping patches with patch_len=16 and stride=8, resulting in 63 patches.
2. WHEN a raw time series of length 512 is provided, THE Patch_Embedding_Module SHALL produce exactly 63 patches using the formula floor((512 - 16) / 8) + 1 = 63.
3. THE Patch_Embedding_Module SHALL implement a PatchEmbedding class containing a linear projection layer that maps each patch of length 16 to an embedding vector of dimension 256.
4. WHEN a batch of time series with shape (batch, seq_len) is provided, THE Patch_Embedding_Module SHALL output a tensor of shape (batch, num_patches, d_model) where d_model is 256.
5. FOR ALL valid input tensors, patching then reconstructing overlapping regions SHALL produce values consistent with the original time series in the overlapping segments (round-trip property for overlap regions).

### Requirement 2: Sinusoidal Positional Encoding

**User Story:** As a developer, I want a positional encoding module that adds order information to patch embeddings using the standard sinusoidal formula, so that the transformer can distinguish patch positions.

#### Acceptance Criteria

1. THE Positional_Encoding_Module SHALL implement sinusoidal positional encoding using the formulas: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)).
2. THE Positional_Encoding_Module SHALL include a comment block showing the sin/cos formula and an ASCII visualization of the encoding pattern.
3. WHEN an embedded patch tensor of shape (batch, num_patches, d_model) is provided, THE Positional_Encoding_Module SHALL return a tensor of the same shape with positional information added.
4. THE Positional_Encoding_Module SHALL register the positional encoding as a non-trainable buffer so it persists across device transfers without consuming optimizer memory.
5. FOR ALL position indices pos1 and pos2 where pos1 is not equal to pos2, THE Positional_Encoding_Module SHALL produce distinct encoding vectors (uniqueness property).

### Requirement 3: Transformer Encoder Block and Stack

**User Story:** As a developer, I want a transformer encoder module with detailed component explanations, so that each architectural choice (attention, LayerNorm, feedforward, residuals) is documented and the encoder can be stacked to 6 layers.

#### Acceptance Criteria

1. THE Transformer_Encoder_Module SHALL implement a TransformerEncoderBlock class containing: multi-head self-attention with 8 heads, pre-norm LayerNorm, a feedforward network with two linear layers and GELU activation (d_model=256 to d_ff=1024 and back), dropout, and residual connections.
2. THE Transformer_Encoder_Module SHALL include explanatory comments for each component describing its role: self-attention for cross-patch pattern detection (seasonality), LayerNorm for training stability, feedforward for independent patch processing, and residual connections for gradient flow.
3. WHEN an input tensor of shape (batch, num_patches, d_model) is provided, THE Transformer_Encoder_Module SHALL return an output tensor of the same shape (batch, num_patches, d_model).
4. THE Transformer_Encoder_Module SHALL support stacking N_LAYERS=6 TransformerEncoderBlock instances sequentially with a final LayerNorm applied after the last block.
5. FOR ALL input tensors, THE Transformer_Encoder_Module SHALL preserve the tensor shape through each layer (shape invariant property).

### Requirement 4: Probabilistic Forecasting Head

**User Story:** As a developer, I want a probabilistic head that predicts three quantiles (P10, P50, P90) per forecast timestep with appropriate quantile loss, so that the model provides calibrated uncertainty estimates.

#### Acceptance Criteria

1. THE Probabilistic_Head_Module SHALL implement a ProbabilisticHead class with three separate linear output heads, one for each quantile (P10, P50, P90).
2. WHEN encoder output of shape (batch, num_patches, d_model) is provided, THE Probabilistic_Head_Module SHALL produce output of shape (batch, forecast_horizon, 3) where the last dimension contains P10, P50, and P90 predictions.
3. THE Probabilistic_Head_Module SHALL implement quantile loss (pinball loss) where the P10 head is penalized more for underestimating and the P90 head is penalized more for overestimating.
4. THE Probabilistic_Head_Module SHALL include explanatory comments describing probabilistic forecasting: P10 as the lower bound (10% chance actual is below), P50 as the median prediction, and P90 as the upper bound (90% chance actual is below).
5. FOR ALL valid inputs, THE Probabilistic_Head_Module SHALL produce outputs where P10 values are less than or equal to P50 values and P50 values are less than or equal to P90 values (monotonicity property).

### Requirement 5: Full PatchTST Model Assembly

**User Story:** As a developer, I want a top-level module that assembles all components into the complete PatchTST model with clear forward() and forecast() methods, so that the model can be used end-to-end for training and inference.

#### Acceptance Criteria

1. THE PatchTST_Assembly_Module SHALL compose the full pipeline: PatchEmbedding, PositionalEncoding, 6 TransformerEncoderBlocks, and ProbabilisticHead in sequence.
2. THE PatchTST_Assembly_Module SHALL implement a forward() method with input/output shape comments that takes a raw time series of shape (batch, seq_len) and returns quantile forecasts of shape (batch, forecast_horizon, 3).
3. THE PatchTST_Assembly_Module SHALL implement a forecast() method that takes a raw time series, applies the forward pass, and returns denormalized P10, P50, and P90 predictions.
4. WHEN initialized, THE PatchTST_Assembly_Module SHALL count and print the total number of trainable parameters, targeting 8-12 million parameters.
5. IF the total parameter count falls outside the range of 8 to 12 million, THEN THE PatchTST_Assembly_Module SHALL print a warning indicating the parameter count is outside the target range.
6. FOR ALL valid input time series, applying forward() then extracting the P50 channel SHALL produce a tensor of shape (batch, forecast_horizon) (round-trip shape property).

### Requirement 6: Quantile Loss Implementation

**User Story:** As a developer, I want a standalone quantile loss function that correctly implements pinball loss for all three quantiles, so that the model can be trained with proper asymmetric penalties.

#### Acceptance Criteria

1. THE Probabilistic_Head_Module SHALL implement a quantile_loss function that computes pinball loss using the formula: loss = tau * max(y - q_hat, 0) + (1 - tau) * max(q_hat - y, 0).
2. WHEN predictions perfectly match targets, THE quantile_loss function SHALL return a loss value of zero.
3. WHEN predictions underestimate targets, THE quantile_loss function SHALL penalize the P90 quantile (tau=0.9) more heavily than the P10 quantile (tau=0.1).
4. WHEN predictions overestimate targets, THE quantile_loss function SHALL penalize the P10 quantile (tau=0.1) more heavily than the P90 quantile (tau=0.9).
5. FOR ALL prediction and target pairs, THE quantile_loss function SHALL return a non-negative scalar value (non-negativity property).

### Requirement 7: Parameter Budget Compliance

**User Story:** As a developer, I want the assembled model to stay within an 8-12M parameter budget suitable for training on a Colab T4 GPU, so that the model is practical for the target hardware.

#### Acceptance Criteria

1. THE PatchTST_Assembly_Module SHALL have a total trainable parameter count between 8 million and 12 million with the default configuration (d_model=256, n_heads=8, n_layers=6, d_ff=1024, patch_len=16, forecast_horizon=96).
2. THE PatchTST_Assembly_Module SHALL provide a count_parameters() method that returns the exact number of trainable parameters.
3. WHEN count_parameters() is called, THE PatchTST_Assembly_Module SHALL return an integer representing the sum of all parameters that require gradient computation.
