# Time Series Foundation Model

A PatchTST-based Time Series Foundation Model built from scratch. The model is pretrained on three domains (Energy, Weather, Finance) using Masked Patch Modeling, then evaluated zero-shot on the ETTh1 benchmark with probabilistic forecasting (P10/P50/P90 quantiles). Deployed as an interactive HuggingFace Space Gradio application.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Dependency Installation](#1-dependency-installation)
  - [2. Data Download](#2-data-download)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Pretraining](#4-pretraining)
  - [5. Zero-Shot Evaluation](#5-zero-shot-evaluation)
  - [6. Baseline Comparison](#6-baseline-comparison)
  - [7. Fine-Tune Evaluation](#7-fine-tune-evaluation)
  - [8. Demo Deployment](#8-demo-deployment)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Testing](#testing)
- [Environment](#environment)

## Project Overview

This project implements a channel-independent patch transformer for time series forecasting. Key design choices:

- **Patch-Based Tokenization**: Time series are segmented into overlapping patches (length 16, stride 8), reducing a 512-step context window to 63 tokens.
- **Masked Patch Modeling**: Self-supervised pretraining masks 40% of patches and reconstructs them, learning temporal representations without labeled data.
- **Probabilistic Output**: A quantile regression head produces P10/P50/P90 forecasts with monotonicity enforcement.
- **Colab-First**: All design choices respect the T4 GPU (15GB VRAM) limit — the model stays under 10M parameters.

## Project Structure

```
Time-Series-Foundation-Model/
├── config.py                    # All hyperparameters (single source of truth)
├── setup.py                     # Pinned dependencies for reproducible installs
├── requirements.txt             # Minimal deps for HuggingFace Space deployment
├── README.md                    # This file — step-by-step pipeline guide
│
├── data/                        # Data acquisition and preprocessing
│   ├── download.py              # Dataset download with retry + verification
│   ├── preprocess.py            # Z-score normalization and chronological splitting
│   ├── patching.py              # Patch creation (length 16, stride 8)
│   ├── dataset.py               # PyTorch Dataset + multi-domain round-robin loader
│   ├── raw/                     # Downloaded CSV files (auto-created)
│   │   └── etth1/              # ETTh1 benchmark dataset
│   └── processed/              # Normalization stats JSON files
│
├── model/                       # PatchTST transformer architecture
│   ├── patch_embedding.py       # Linear projection (16→256) + positional encoding
│   ├── attention.py             # Multi-head self-attention (8 heads, scaled dot-product)
│   ├── transformer_layer.py     # Pre-norm encoder layer (LN→MHSA→Res→LN→FFN→Res)
│   ├── encoder.py               # 6-layer transformer encoder stack
│   └── patchtst.py              # Top-level model: patch embedding + encoder
│
├── pretraining/                 # Masked Patch Modeling self-supervised training
│   ├── masking.py               # Random 40% patch masking with learnable mask token
│   ├── reconstruction_head.py   # Linear projection (256→16) + masked MSE loss
│   └── train.py                 # Full pretraining loop (multi-domain, checkpointing)
│
├── forecasting/                 # Probabilistic forecasting and fine-tuning
│   ├── probabilistic_head.py    # Quantile regression head (P10/P50/P90) + pinball loss
│   ├── inference.py             # Zero-shot inference with sliding window
│   └── finetune.py              # Fine-tune head on ETTh1 (encoder frozen)
│
├── evaluation/                  # Metrics, baselines, and visualization
│   ├── metrics.py               # MAE, MSE, MASE, CRPS computation
│   ├── baselines.py             # ARIMA, Prophet, seasonal naive fallback
│   ├── evaluate.py              # Full evaluation pipeline + results table
│   └── visualize.py             # Forecast plots with prediction intervals
│
├── utils/                       # Colab utilities and experiment logging
│   ├── colab_helpers.py         # Drive mount, checkpointing, VRAM monitoring
│   └── logger.py                # W&B logging with CSV fallback
│
├── app/                         # Interactive demo application
│   └── app.py                   # Gradio app (HuggingFace Space entry point)
│
├── checkpoints/                 # Saved model checkpoints (.pt files)
│
└── tests/                       # Test suite
    ├── unit/                    # Unit tests for individual components
    ├── properties/              # Property-based tests (Hypothesis)
    └── integration/             # End-to-end pipeline tests
```

### How Modules Relate

The pipeline flows in stages, with each module consuming the output of the previous:

1. **`config.py`** is imported by every module — it defines all hyperparameters in one place.
2. **`data/download.py`** fetches raw CSVs → **`data/preprocess.py`** normalizes and splits them → **`data/patching.py`** segments into patches → **`data/dataset.py`** wraps them as PyTorch Datasets.
3. **`model/`** builds the PatchTST encoder bottom-up: `patch_embedding.py` → `attention.py` → `transformer_layer.py` → `encoder.py` → `patchtst.py`.
4. **`pretraining/train.py`** uses the model + `masking.py` + `reconstruction_head.py` to pretrain on multi-domain data.
5. **`forecasting/inference.py`** attaches the `probabilistic_head.py` to the pretrained encoder for zero-shot forecasting.
6. **`forecasting/finetune.py`** freezes the encoder and trains only the forecast head on ETTh1.
7. **`evaluation/evaluate.py`** orchestrates zero-shot + baselines + metrics + visualization.
8. **`app/app.py`** loads the pretrained model and serves an interactive Gradio demo.
9. **`utils/`** provides cross-cutting concerns: checkpointing (`colab_helpers.py`) and logging (`logger.py`).

## Quick Start

### 1. Dependency Installation

Install all project dependencies with exact pinned versions:

```bash
pip install -e .
```

This uses `setup.py` to install all required packages (PyTorch, NumPy, Pandas, Matplotlib, Gradio, W&B, Hypothesis, etc.) with pinned versions for reproducibility.

Alternatively, for the minimal runtime dependencies only (e.g., for HuggingFace Space):

```bash
pip install -r requirements.txt
```

### 2. Data Download

Download all four datasets (Energy, Weather, Finance, ETTh1) with automatic retry and verification:

```bash
python -m data.download
```

This will:
- Download each dataset CSV to `data/raw/` (ETTh1 goes to `data/raw/etth1/`)
- Retry up to 3 times with exponential backoff on network failures
- Verify each file has a valid header and at least 1,000 rows
- Print statistics (row count, column count, date range, missing %, file size)
- Skip downloads if files already exist

### 3. Data Preprocessing

Preprocess the downloaded data (normalize, split, save stats):

```python
python -c "
from data.preprocess import preprocess_dataset
import pandas as pd
import numpy as np

# Preprocess each dataset
for name, path in [
    ('energy', 'data/raw/energy.csv'),
    ('weather', 'data/raw/weather.csv'),
    ('finance', 'data/raw/finance.csv'),
    ('etth1', 'data/raw/etth1/etth1.csv'),
]:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    raw_data = df[numeric_cols].values.astype(np.float64)
    train, val, test, stats = preprocess_dataset(raw_data, name)
    print(f'{name}: train={train.shape}, val={val.shape}, test={test.shape}')
"
```

This will:
- Split each dataset chronologically into train (70%), validation (15%), test (15%)
- Compute per-channel mean/std from the training split only (no data leakage)
- Normalize all splits using training statistics
- Save normalization stats as JSON to `data/processed/`

### 4. Pretraining

Pretrain the PatchTST model using Masked Patch Modeling across three domains:

```python
python -c "
import torch
import numpy as np
import pandas as pd
from config import Config
from model.patchtst import PatchTSTModel
from data.dataset import TimeSeriesDataset
from data.preprocess import preprocess_dataset
from pretraining.train import pretrain

# Load and preprocess training data for each domain
datasets_train = []
datasets_val = []
for name, path in [
    ('energy', 'data/raw/energy.csv'),
    ('weather', 'data/raw/weather.csv'),
    ('finance', 'data/raw/finance.csv'),
]:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    raw = df[numeric_cols[0]].values.astype(np.float64)
    train, val, test, stats = preprocess_dataset(raw, name)
    datasets_train.append(TimeSeriesDataset(train, Config.CONTEXT_LENGTH, Config.FORECAST_HORIZON))
    datasets_val.append(TimeSeriesDataset(val, Config.CONTEXT_LENGTH, Config.FORECAST_HORIZON))

# Initialize model and run pretraining
model = PatchTSTModel(Config)
print(f'Model parameters: {model.count_parameters():,}')

history = pretrain(
    model=model,
    train_datasets=datasets_train,
    val_datasets=datasets_val,
)
print(f'Pretraining complete. Epochs: {history[\"epochs_completed\"]}')
"
```

Key pretraining settings (from `config.py`):
- 20 epochs, batch size 32, gradient accumulation 4 (effective batch 128)
- AdamW optimizer, lr=1e-4, weight decay=0.01
- Cosine LR schedule with 2-epoch linear warmup, decaying to 1e-6
- 40% random patch masking with MSE reconstruction loss
- Checkpoints saved to Google Drive every epoch

### 5. Zero-Shot Evaluation

Evaluate the pretrained model on ETTh1 without any fine-tuning:

```bash
python -m evaluation.evaluate
```

This will:
- Load the pretrained checkpoint from `checkpoints/pretrained_patchtst.pt`
- Generate P10/P50/P90 probabilistic forecasts on ETTh1 test windows
- Use sliding window: context=512, horizon=96, stride=96 (non-overlapping)
- Compute MAE, MSE, MASE (using P50), and CRPS (using all quantiles)
- Print a formatted comparison table
- Generate visualization plots saved to `evaluation/`

### 6. Baseline Comparison

Baselines (ARIMA, Prophet) are run automatically as part of the evaluation pipeline above. To run baselines independently:

```python
python -c "
import numpy as np
import pandas as pd
from data.preprocess import split_chronological, compute_normalization_stats, normalize, inverse_normalize
from evaluation.baselines import run_all_baselines

# Load ETTh1 data
df = pd.read_csv('data/raw/etth1/etth1.csv')
raw = df.select_dtypes(include=[np.number]).iloc[:, 0].values.astype(np.float64)
train, val, test = split_chronological(raw)

# Run ARIMA and Prophet baselines on the test split
results = run_all_baselines(
    train=train,
    test_data=test,
    context_length=512,
    forecast_horizon=96,
    stride=96,
)

for name, metrics in results.items():
    print(f'{name}: MAE={metrics[\"mae\"]:.4f}, MSE={metrics[\"mse\"]:.4f}, MASE={metrics[\"mase\"]:.4f}')
"
```

Baselines include:
- **ARIMA**: Auto order selection (max p=5, d=2, q=5)
- **Prophet**: Facebook Prophet point forecasts
- **Seasonal Naive**: Fallback (period=24) if ARIMA/Prophet fail to converge

### 7. Fine-Tune Evaluation

Fine-tune the pretrained model on ETTh1 (encoder frozen, only forecast head trained):

```bash
python -m forecasting.finetune
```

This will:
- Load pretrained encoder weights from `checkpoints/pretrained_patchtst.pt`
- Freeze all encoder parameters
- Train only the ProbabilisticForecastHead for 10 epochs (lr=1e-5, batch=32)
- Use cosine LR schedule with 1-epoch warmup
- Select best checkpoint by validation loss
- Apply early stopping if NaN or 3 consecutive val loss increases
- Evaluate on test split and print MAE, MSE, CRPS
- Save fine-tuned checkpoint to `checkpoints/finetuned_patchtst.pt`

### 8. Demo Deployment

Launch the interactive Gradio forecasting demo locally:

```bash
python app/app.py
```

The app will be available at `http://localhost:7860`. Features:
- Upload a CSV file (≥512 rows, with a datetime column and numeric columns)
- Select a target column from a dropdown
- Choose forecast horizon (24–192 steps)
- View probabilistic forecast plot (P50 line + P10–P90 shaded interval)

To deploy as a HuggingFace Space:
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Push this repository (with `app/app.py` as the entry point)
3. The Space uses `requirements.txt` for dependency installation
4. Place a pretrained checkpoint in `checkpoints/` for the model to load

## Architecture

The PatchTST model processes time series as follows:

```
Input: (batch, 512)                    # Raw univariate time series
  → Patching: (batch, 63, 16)          # 63 overlapping patches of length 16
  → Linear Projection: (batch, 63, 256) # Project each patch to d_model=256
  → + Positional Encoding              # Add learnable position embeddings
  → 6× Transformer Encoder Layers      # Pre-norm: LN→MHSA(8 heads)→Res→LN→FFN→Res
  → Encoder Output: (batch, 63, 256)   # Contextualized patch embeddings
  → Probabilistic Head: (batch, 96, 3) # P10/P50/P90 quantile forecasts
```

Total parameters: ~9.4M (well under the 10M budget for T4 GPU).

## Configuration

All hyperparameters are centralized in `config.py`. Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| D_MODEL | 256 | Transformer hidden dimension |
| N_HEADS | 8 | Attention heads |
| N_LAYERS | 6 | Transformer encoder layers |
| PATCH_LEN | 16 | Patch length (time steps) |
| PATCH_STRIDE | 8 | Stride between patches |
| CONTEXT_LENGTH | 512 | Input context window |
| FORECAST_HORIZON | 96 | Prediction horizon |
| MASK_RATIO | 0.4 | Pretraining mask ratio |
| PRETRAIN_LR | 1e-4 | Pretraining learning rate |
| PRETRAIN_EPOCHS | 20 | Pretraining epochs |
| QUANTILES | [0.1, 0.5, 0.9] | P10/P50/P90 output quantiles |

## Testing

Run the full test suite:

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run property-based tests (Hypothesis, 100 iterations per property)
pytest tests/properties/ -v --hypothesis-show-statistics

# Run integration tests
pytest tests/integration/ -v
```

The test suite validates 18 correctness properties covering normalization round-trips, patch dimensions, model output shapes, quantile monotonicity, metric relationships, and more.

## Environment

This project is designed for **Google Colab free tier** with a T4 GPU (~15GB VRAM). Key constraints:

- Model fits in ~12.6GB VRAM (with ~2.4GB headroom)
- Gradient accumulation (4 steps) achieves effective batch size 128
- Checkpoints are saved to Google Drive for session recovery
- Session timer warns after 10 hours of runtime
- W&B logging falls back to local CSV if no API key is configured
