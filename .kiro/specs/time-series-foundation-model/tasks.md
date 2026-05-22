# Implementation Plan: Time Series Foundation Model

## Overview

This plan implements a PatchTST-based Time Series Foundation Model from scratch in Python. The implementation follows a modular structure (one file per component), pretrains on 3 domains using Masked Patch Modeling, evaluates zero-shot on ETTh1 with probabilistic forecasting (P10/P50/P90), and deploys as a HuggingFace Space Gradio app. All code targets Google Colab free tier (T4 GPU, ~15GB VRAM) and is heavily commented for beginners.

## Tasks

- [x] 1. Set up project structure, configuration, and dependencies
  - [x] 1.1 Create `config.py` with all hyperparameters
    - Define the `Config` class with all constants: D_MODEL=256, N_HEADS=8, N_LAYERS=6, D_FF=1024, DROPOUT=0.1, PATCH_LEN=16, PATCH_STRIDE=8, CONTEXT_LENGTH=512, NUM_PATCHES=63, MASK_RATIO=0.4, PRETRAIN_LR=1e-4, PRETRAIN_EPOCHS=20, PRETRAIN_BATCH_SIZE=32, GRADIENT_ACCUMULATION=4, WEIGHT_DECAY=0.01, WARMUP_EPOCHS=2, MIN_LR=1e-6, FORECAST_HORIZON=96, QUANTILES=[0.1, 0.5, 0.9], FINETUNE_LR=1e-5, FINETUNE_EPOCHS=10, FINETUNE_BATCH_SIZE=32, TRAIN_RATIO=0.70, VAL_RATIO=0.15, TEST_RATIO=0.15, MAX_RETRIES=3, RETRY_BASE_DELAY=2.0
    - Include module-level docstring and inline comments explaining each group of parameters
    - _Requirements: 1.3, 1.5_

  - [x] 1.2 Create `setup.py` with pinned dependencies
    - Pin all dependencies with exact versions (`==`): torch, numpy, pandas, matplotlib, scikit-learn, hypothesis, pytest, gradio, wandb, statsmodels, prophet, huggingface_hub, tqdm
    - Include module-level docstring explaining the file's purpose
    - _Requirements: 1.4, 1.5_

  - [x] 1.3 Create `requirements.txt` for HuggingFace Space deployment
    - List runtime dependencies needed for the Gradio app (torch, numpy, pandas, matplotlib, gradio)
    - _Requirements: 13.4_

  - [x] 1.4 Create directory structure and `__init__.py` files
    - Create directories: `data/`, `data/raw/`, `data/raw/etth1/`, `data/processed/`, `model/`, `pretraining/`, `forecasting/`, `evaluation/`, `utils/`, `app/`, `tests/`, `tests/properties/`, `tests/unit/`, `tests/integration/`
    - Add `__init__.py` with module-level docstrings in each package directory
    - _Requirements: 1.1, 1.2, 1.5_

- [x] 2. Implement data acquisition pipeline
  - [x] 2.1 Implement `data/download.py` — dataset download with retry logic
    - Implement `download_dataset(name, url, save_dir)` with exponential backoff retry (3 attempts, starting at 2s)
    - Implement `verify_dataset(filepath, min_rows)` to check header and minimum 1000 rows
    - Implement `download_all()` to download Energy, Weather, Finance, and ETTh1 datasets
    - Skip download if file already exists; delete corrupt files on verification failure
    - Print dataset statistics (row count, column count, date range, missing %, file size)
    - Include heavy inline comments and type hints on all functions
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 2.2 Write unit tests for `data/download.py`
    - Test retry logic with mocked network failures
    - Test skip-existing-file behavior
    - Test verification rejects files with too few rows
    - Test corrupt file deletion on verification failure
    - _Requirements: 2.6, 2.7, 2.8_

- [x] 3. Implement data preprocessing and patching
  - [x] 3.1 Implement `data/preprocess.py` — normalization and splitting
    - Implement `compute_normalization_stats(train_data)` for per-channel mean/std
    - Implement `normalize(data, stats)` and `inverse_normalize(data, stats)` for z-score normalization
    - Implement `split_chronological(data)` for 70/15/15 train/val/test split preserving time order
    - Handle zero-std channels by setting std=1.0 with warning
    - Save normalization stats as JSON to `data/processed/`
    - Filter out series shorter than 512 time steps with warning
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6, 14.1, 14.2, 14.3, 14.5_

  - [x] 3.2 Implement `data/patching.py` — patch creation logic
    - Implement `create_patches(series, patch_len=16, stride=8)` producing overlapping patches
    - Implement `compute_num_patches(series_length, patch_len, stride)` returning floor((L-16)/8)+1
    - Discard trailing steps that don't form a complete patch
    - _Requirements: 3.2, 3.7, 14.1, 14.2, 14.3, 14.5_

  - [x] 3.3 Implement `data/dataset.py` — PyTorch Dataset and multi-domain loader
    - Implement `TimeSeriesDataset` class yielding (context_window, target) pairs
    - Implement `MultiDomainDataLoader` for round-robin interleaved batching across 3 domains
    - _Requirements: 5.3, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 3.4 Write property test for normalization round-trip
    - **Property 1: Normalization Round-Trip**
    - Test that normalize followed by inverse_normalize recovers original values within 1e-6 tolerance
    - Use Hypothesis strategies to generate random float arrays
    - **Validates: Requirements 3.1, 6.4**

  - [ ]* 3.5 Write property test for patch count and dimensions
    - **Property 2: Patch Count and Dimensions**
    - Test that create_patches produces exactly floor((L-16)/8)+1 patches of shape (patch_len,)
    - Use Hypothesis to generate series of varying lengths (16 to 2048)
    - **Validates: Requirements 3.2, 3.7**

  - [ ]* 3.6 Write property test for chronological split preservation
    - **Property 3: Chronological Split Preservation**
    - Test that train/val/test splits are non-overlapping, contiguous, and concatenate to original
    - Use Hypothesis to generate arrays of varying lengths
    - **Validates: Requirements 3.3**

  - [ ]* 3.7 Write property test for normalization statistics serialization
    - **Property 4: Normalization Statistics Serialization Round-Trip**
    - Test that serializing stats to JSON and deserializing produces identical dictionary
    - Use Hypothesis to generate dictionaries with float mean/std values
    - **Validates: Requirements 3.4**

  - [ ]* 3.8 Write property test for short series filtering
    - **Property 5: Short Series Filtering**
    - Test that only series with length >= 512 are retained, preserving order
    - Use Hypothesis to generate lists of series with varying lengths
    - **Validates: Requirements 3.5**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement PatchTST model architecture
  - [x] 5.1 Implement `model/patch_embedding.py` — patch projection and positional encoding
    - Implement `PatchEmbedding` module: linear projection from patch_len (16) to d_model (256)
    - Add learnable positional encodings of dimension 256 for 63 patch positions
    - Include heavy inline comments explaining each operation
    - _Requirements: 4.1, 4.2, 14.1, 14.2, 14.3, 14.5_

  - [x] 5.2 Implement `model/attention.py` — multi-head self-attention
    - Implement `MultiHeadSelfAttention` with 8 heads, scaled dot-product attention
    - Apply dropout of 0.1 after attention
    - _Requirements: 4.3, 4.4, 14.1, 14.2, 14.3, 14.5_

  - [x] 5.3 Implement `model/transformer_layer.py` — single encoder layer
    - Implement `TransformerEncoderLayer` with pre-norm architecture (LN → MHSA → Residual → LN → FFN → Residual)
    - FFN: 256 → 1024 → 256 with GELU activation
    - Apply dropout of 0.1 after attention and FFN sublayers
    - _Requirements: 4.3, 4.4, 4.5, 14.1, 14.2, 14.3, 14.5_

  - [x] 5.4 Implement `model/encoder.py` — full 6-layer encoder stack
    - Implement `PatchTSTEncoder` stacking 6 transformer layers with final layer norm
    - _Requirements: 4.3, 14.1, 14.2, 14.3, 14.5_

  - [x] 5.5 Implement `model/patchtst.py` — top-level model assembly
    - Implement `PatchTSTModel` combining patch embedding + encoder (channel-independent)
    - Input: (batch, 512) → Output: (batch, 63, 256)
    - Implement `count_parameters()` method; verify < 10M parameters
    - Raise ValueError for inputs shorter than minimum required length
    - _Requirements: 4.1, 4.6, 4.7, 4.8, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 5.6 Write property test for model output shape invariant
    - **Property 6: Model Output Shape Invariant**
    - Test that input (B, 512) always produces output (B, 63, 256) for any B >= 1
    - Use Hypothesis to generate batch sizes from 1 to 8
    - **Validates: Requirements 4.1, 4.6**

  - [ ]* 5.7 Write property test for invalid input rejection
    - **Property 7: Invalid Input Rejection**
    - Test that inputs with length < 16 raise ValueError with appropriate message
    - Use Hypothesis to generate lengths from 1 to 15
    - **Validates: Requirements 4.8**

- [x] 6. Implement masked patch modeling pretraining
  - [x] 6.1 Implement `pretraining/masking.py` — random patch masking
    - Implement `PatchMasker` class with learnable mask token of dimension 256
    - Randomly select 40% of patches per sample (uniform, without replacement)
    - Replace selected patches with the mask token
    - Return masked embeddings and mask indices
    - _Requirements: 5.1, 14.1, 14.2, 14.3, 14.5_

  - [x] 6.2 Implement `pretraining/reconstruction_head.py` — MSE reconstruction
    - Implement `ReconstructionHead`: linear projection from d_model (256) to patch_len (16)
    - Compute MSE loss only over masked positions
    - _Requirements: 5.2, 14.1, 14.2, 14.3, 14.5_

  - [x] 6.3 Implement `pretraining/train.py` — full pretraining loop
    - Multi-domain round-robin batching (Energy, Weather, Finance)
    - AdamW optimizer: lr=1e-4, weight_decay=0.01, gradient accumulation over 4 steps
    - Cosine LR schedule decaying to 1e-6 with linear warmup over first 2 epochs
    - Train for 20 epochs; save checkpoint to Google Drive every epoch
    - Compute and log validation loss across all 3 domains each epoch
    - Handle NaN/divergence: stop training, save last valid checkpoint
    - Handle step timeout (>20min): warning + auto-checkpoint
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 6.4 Write property test for mask ratio invariant
    - **Property 8: Mask Ratio Invariant**
    - Test that masking module masks exactly round(0.4 * N) patches per sample
    - Use Hypothesis to generate batch sizes and verify mask counts
    - **Validates: Requirements 5.1**

  - [ ]* 6.5 Write property test for masked-only loss computation
    - **Property 9: Masked-Only Loss Computation**
    - Test that reconstruction loss equals MSE only over masked positions, ignoring unmasked
    - Use Hypothesis to generate random predictions, targets, and masks
    - **Validates: Requirements 5.2**

- [x] 7. Implement probabilistic forecasting head
  - [x] 7.1 Implement `forecasting/probabilistic_head.py` — quantile regression head
    - Implement `ProbabilisticForecastHead`: maps (B, 63, 256) → (B, 96, 3)
    - Output P10/P50/P90 quantiles with monotonicity enforcement (sort at each step)
    - Implement `quantile_loss` (pinball loss) for quantile levels [0.1, 0.5, 0.9]
    - Raise error if input embeddings don't correspond to valid context window
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 14.1, 14.2, 14.3, 14.5_

  - [x] 7.2 Implement `forecasting/inference.py` — zero-shot inference
    - Implement `zero_shot_forecast`: load pretrained model, no gradient computation, no dropout
    - Generate probabilistic forecasts and apply inverse normalization to return original scale
    - Use sliding window: context=512, horizon=96, stride=96 (non-overlapping forecasts)
    - _Requirements: 7.1, 7.2, 6.4, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 7.3 Write property test for probabilistic head output shape
    - **Property 10: Probabilistic Head Output Shape**
    - Test that encoder output (B, 63, 256) always produces (B, 96, 3) output
    - Use Hypothesis to generate batch sizes from 1 to 8
    - **Validates: Requirements 6.1**

  - [ ]* 7.4 Write property test for pinball loss correctness
    - **Property 11: Pinball Loss Correctness**
    - Test that pinball loss equals tau * max(y - q_hat, 0) + (1-tau) * max(q_hat - y, 0)
    - Use Hypothesis to generate random predictions, targets, and quantile levels
    - **Validates: Requirements 6.2**

  - [ ]* 7.5 Write property test for quantile monotonicity
    - **Property 12: Quantile Monotonicity**
    - Test that output satisfies P10 <= P50 <= P90 at every position
    - Use Hypothesis to generate random encoder outputs and verify sorted order
    - **Validates: Requirements 6.3**

  - [ ]* 7.6 Write property test for sliding window count
    - **Property 13: Sliding Window Count**
    - Test that number of evaluation windows equals floor((T - 512 - 96) / 96) + 1
    - Use Hypothesis to generate test set lengths and verify window count
    - **Validates: Requirements 7.2**

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement evaluation pipeline
  - [x] 9.1 Implement `evaluation/metrics.py` — MAE, MSE, MASE, CRPS
    - Implement `mae`, `mse`, `mase` (with seasonal_period=24), and `crps_quantile` functions
    - All metrics computed on ETTh1 test split using P50 for point metrics, all quantiles for CRPS
    - _Requirements: 9.1, 14.1, 14.2, 14.3, 14.5_

  - [x] 9.2 Implement `evaluation/baselines.py` — ARIMA and Prophet baselines
    - Implement `run_arima_baseline` with auto ARIMA (max p=5, d=2, q=5)
    - Implement `run_prophet_baseline` for point forecasts
    - Implement `seasonal_naive_fallback` (period=24) as fallback for convergence failures
    - Compute MAE, MSE, MASE for each baseline on same test windows
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 14.1, 14.2, 14.3, 14.5_

  - [x] 9.3 Implement `evaluation/evaluate.py` — full evaluation pipeline
    - Orchestrate zero-shot evaluation, baseline comparison, and results table
    - Print formatted comparison table: PatchTST zero-shot, PatchTST fine-tuned, ARIMA, Prophet
    - Columns: MAE, MSE, MASE, CRPS (rounded to 4 decimal places)
    - Raise error if pretrained checkpoint not found
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 9.2, 14.1, 14.2, 14.3, 14.5_

  - [x] 9.4 Implement `evaluation/visualize.py` — plotting with prediction intervals
    - Generate plots: actual values, P50 predictions, shaded P10-P90 intervals
    - Plot at least 5 test windows evenly spaced across ETTh1 test split
    - Include axis labels, descriptive titles, and legends
    - Display inline in Colab; save as PNG at 150+ DPI to `evaluation/` directory
    - _Requirements: 9.3, 9.4, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 9.5 Write property test for seasonal naive periodicity
    - **Property 14: Seasonal Naive Periodicity**
    - Test that forecast[t] == history[-(p - (t % p))] for all t in [0, h)
    - Use Hypothesis to generate history arrays and forecast horizons
    - **Validates: Requirements 8.4, 8.5**

  - [ ]* 9.6 Write property test for metric non-negativity and relationships
    - **Property 15: Metric Non-Negativity and Relationships**
    - Test MAE >= 0, MSE >= 0, MAE <= sqrt(MSE * n) / n, and zero metrics when predictions == targets
    - Use Hypothesis to generate random prediction/target arrays
    - **Validates: Requirements 9.1**

- [x] 10. Implement Colab utilities and experiment logging
  - [x] 10.1 Implement `utils/colab_helpers.py` — Drive mount, checkpoints, VRAM monitoring
    - Implement `mount_drive()`: mount Google Drive, create checkpoint directory
    - Implement `save_checkpoint(model, optimizer, epoch, loss, max_keep=5)`: save with timestamp, retain max 5
    - Implement `load_checkpoint(model, optimizer)`: load most recent, return None if none exists
    - Implement `check_vram()`: print GPU memory usage, return True if >= 2GB free
    - Implement `session_timer()`: return elapsed minutes, warn if > 10 hours
    - Handle IOError on save failure; handle no-GPU case
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 14.1, 14.2, 14.3, 14.5_

  - [x] 10.2 Implement `utils/logger.py` — W&B logging with CSV fallback
    - Implement `ExperimentLogger` class: init W&B run with project name, hyperparams, and run name
    - Log training loss, validation loss, learning rate, epoch, GPU memory each epoch
    - Fall back to local CSV if W&B not configured (no API key)
    - Handle W&B logging failures gracefully (write to CSV, continue training)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 10.3 Write property test for checkpoint retention limit
    - **Property 16: Checkpoint Retention Limit**
    - Test that after N saves with max_keep=5, exactly min(N, 5) files remain (most recent)
    - Use Hypothesis to generate sequences of save operations
    - **Validates: Requirements 10.2**

- [x] 11. Implement fine-tune evaluation
  - [x] 11.1 Implement `forecasting/finetune.py` — fine-tune on ETTh1
    - Load pretrained weights, freeze encoder, train only ProbabilisticHead
    - Train 10 epochs, batch_size=32, lr=1e-5, cosine schedule with warmup over first epoch
    - Select best checkpoint by validation loss; compute MAE, MSE, CRPS on test split
    - Early stopping: stop if NaN loss or 3 consecutive val loss increases; restore best checkpoint
    - Include results in comparison table
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 11.2 Write property test for early stopping detection
    - **Property 17: Early Stopping Detection**
    - Test that early stopping triggers iff 3 consecutive val loss increases or NaN detected
    - Use Hypothesis to generate sequences of validation loss values
    - **Validates: Requirements 12.5**

- [x] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Gradio demo application
  - [x] 13.1 Implement `app/app.py` — interactive forecasting demo
    - File upload widget accepting CSV up to 50MB
    - Populate dropdown with numeric columns; slider for forecast horizon (24-192)
    - Load pretrained model, run inference on last 512 steps of selected column
    - Display plot: historical context, P50 forecast line, shaded P10-P90 region
    - Complete inference within 30 seconds
    - Error handling: reject < 512 rows, reject no numeric/datetime columns, handle inference timeout
    - Allow retry without re-uploading on failure
    - Deployable as HuggingFace Space with `app.py` as entry point
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 14.1, 14.2, 14.3, 14.5_

  - [ ]* 13.2 Write property test for CSV input validation
    - **Property 18: CSV Input Validation**
    - Test: reject files < 512 rows, correctly identify numeric columns, reject files with zero numeric or no datetime column
    - Use Hypothesis to generate DataFrames with varying column types and row counts
    - **Validates: Requirements 13.2, 13.5, 13.6**

- [x] 14. Create README with step-by-step guide
  - [x] 14.1 Write `README.md` with complete pipeline documentation
    - Include step-by-step guide for: dependency installation, data download, preprocessing, pretraining, zero-shot evaluation, baseline comparison, fine-tune evaluation, demo deployment
    - Provide exact commands for each stage
    - Explain project structure and how modules relate
    - _Requirements: 14.4_

- [x] 15. Final checkpoint — Ensure all tests pass and integration works
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 18 universal correctness properties defined in the design
- Unit tests validate specific examples and edge cases
- All code must include heavy inline comments for beginner readability (Requirement 14)
- All modules must stay under 300 lines of code excluding comments (Requirement 1.2)
- Target environment is Google Colab free tier with T4 GPU (~15GB VRAM)
- Python is the implementation language; PyTorch for deep learning, Hypothesis for property-based testing

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2"] },
    { "id": 2, "tasks": ["2.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"] },
    { "id": 3, "tasks": ["5.1", "5.2", "5.3"] },
    { "id": 4, "tasks": ["5.4", "5.5"] },
    { "id": 5, "tasks": ["5.6", "5.7", "6.1", "6.2"] },
    { "id": 6, "tasks": ["6.3", "6.4", "6.5"] },
    { "id": 7, "tasks": ["7.1", "7.2"] },
    { "id": 8, "tasks": ["7.3", "7.4", "7.5", "7.6"] },
    { "id": 9, "tasks": ["9.1", "9.2", "9.4", "10.1", "10.2"] },
    { "id": 10, "tasks": ["9.3", "9.5", "9.6", "10.3"] },
    { "id": 11, "tasks": ["11.1"] },
    { "id": 12, "tasks": ["11.2", "13.1"] },
    { "id": 13, "tasks": ["13.2", "14.1"] }
  ]
}
```
