"""Central configuration module for the Time Series Foundation Model.

This file defines all hyperparameters used across the entire pipeline — model
architecture, pretraining, forecasting, fine-tuning, data processing, and
reliability settings. Every other module imports from this single source of
truth, making it easy to adjust experiments without hunting through code.

Related modules:
    - model/ uses architecture params (D_MODEL, N_HEADS, etc.)
    - pretraining/ uses training params (PRETRAIN_LR, MASK_RATIO, etc.)
    - forecasting/ uses forecast params (FORECAST_HORIZON, QUANTILES, etc.)
    - data/ uses data params (TRAIN_RATIO, PATCH_LEN, CONTEXT_LENGTH, etc.)
    - utils/ uses reliability params (MAX_RETRIES, RETRY_BASE_DELAY)
"""


class Config:
    """All hyperparameters for the Time Series Foundation Model.

    This class stores every tunable constant as a class-level attribute.
    Import and reference directly: `from config import Config` then use
    `Config.D_MODEL`, `Config.PRETRAIN_LR`, etc.

    Attributes are grouped by function: model architecture, patching,
    pretraining, forecasting, fine-tuning, data splitting, and reliability.
    """

    # -------------------------------------------------------------------------
    # Model Architecture
    # These define the transformer encoder dimensions and regularization.
    # -------------------------------------------------------------------------

    # Hidden dimension of the transformer (embedding size for each patch)
    D_MODEL: int = 256

    # Number of attention heads in multi-head self-attention
    N_HEADS: int = 8

    # Number of stacked transformer encoder layers
    N_LAYERS: int = 6

    # Feed-forward network inner dimension (4 * D_MODEL)
    D_FF: int = 1024

    # Dropout probability applied after attention and FFN sublayers
    DROPOUT: float = 0.1

    # -------------------------------------------------------------------------
    # Patching Parameters
    # Control how raw time series are segmented into fixed-length patches
    # before being fed to the transformer.
    # -------------------------------------------------------------------------

    # Length of each patch (number of time steps per patch)
    PATCH_LEN: int = 16

    # Stride between consecutive patches (overlap = PATCH_LEN - PATCH_STRIDE)
    PATCH_STRIDE: int = 8

    # Number of input time steps the model consumes (context window)
    CONTEXT_LENGTH: int = 512

    # Number of patches produced: floor((CONTEXT_LENGTH - PATCH_LEN) / PATCH_STRIDE) + 1
    NUM_PATCHES: int = 63

    # -------------------------------------------------------------------------
    # Pretraining Parameters
    # Settings for Masked Patch Modeling self-supervised pretraining across
    # three domains (Energy, Weather, Finance).
    # -------------------------------------------------------------------------

    # Fraction of patches masked per sample during pretraining
    MASK_RATIO: float = 0.4

    # Base learning rate for the AdamW optimizer during pretraining
    PRETRAIN_LR: float = 1e-4

    # Total number of pretraining epochs
    PRETRAIN_EPOCHS: int = 20

    # Batch size per gradient accumulation step
    PRETRAIN_BATCH_SIZE: int = 32

    # Number of forward passes before a single optimizer step (effective batch = 128)
    GRADIENT_ACCUMULATION: int = 4

    # L2 regularization coefficient for AdamW
    WEIGHT_DECAY: float = 0.01

    # Number of epochs for linear learning rate warmup
    WARMUP_EPOCHS: int = 2

    # Minimum learning rate at the end of cosine decay schedule
    MIN_LR: float = 1e-6

    # -------------------------------------------------------------------------
    # Forecasting Parameters
    # Define the prediction horizon and quantile levels for probabilistic output.
    # -------------------------------------------------------------------------

    # Number of future time steps the model predicts
    FORECAST_HORIZON: int = 96

    # Quantile levels for probabilistic forecasting (P10, P50, P90)
    QUANTILES: list = [0.1, 0.5, 0.9]

    # -------------------------------------------------------------------------
    # Fine-Tuning Parameters
    # Settings for fine-tuning the forecast head on ETTh1 (encoder frozen).
    # -------------------------------------------------------------------------

    # Learning rate for fine-tuning (lower than pretraining to preserve features)
    FINETUNE_LR: float = 1e-5

    # Number of fine-tuning epochs on the target dataset
    FINETUNE_EPOCHS: int = 10

    # Batch size during fine-tuning
    FINETUNE_BATCH_SIZE: int = 32

    # -------------------------------------------------------------------------
    # Data Splitting Ratios
    # Chronological split proportions for train/validation/test sets.
    # -------------------------------------------------------------------------

    # Fraction of data used for training
    TRAIN_RATIO: float = 0.70

    # Fraction of data used for validation (hyperparameter tuning)
    VAL_RATIO: float = 0.15

    # Fraction of data used for final evaluation
    TEST_RATIO: float = 0.15

    # -------------------------------------------------------------------------
    # Reliability and Retry Settings
    # Control retry behavior for network operations (dataset downloads).
    # -------------------------------------------------------------------------

    # Maximum number of download retry attempts before raising an error
    MAX_RETRIES: int = 3

    # Initial delay in seconds for exponential backoff (doubles each retry)
    RETRY_BASE_DELAY: float = 2.0

    # -------------------------------------------------------------------------
    # Enhanced Pretraining Parameters
    # Settings for multi-task pretraining with domain classification,
    # step-based logging/checkpointing, early stopping, and model export.
    # -------------------------------------------------------------------------

    # Per-domain sampling weights for DomainMixedDataLoader (must sum to 1.0)
    DOMAIN_WEIGHTS: dict = {"energy": 0.4, "weather": 0.3, "finance": 0.3}

    # Weight applied to domain classification loss in multi-task total loss
    DOMAIN_LOSS_WEIGHT: float = 0.1

    # Number of pretraining domains (Energy, Weather, Finance)
    NUM_DOMAINS: int = 3

    # Log metrics to W&B (or stdout fallback) every N optimizer steps
    LOG_EVERY_N_STEPS: int = 50

    # Save a checkpoint every N optimizer steps
    CHECKPOINT_EVERY_N_STEPS: int = 500

    # Maximum number of checkpoint files to retain (oldest deleted first)
    MAX_CHECKPOINTS: int = 5

    # Number of epochs without improvement before early stopping triggers
    EARLY_STOPPING_PATIENCE: int = 5

    # Minimum validation loss decrease to qualify as improvement
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # Weights & Biases project name for logging
    WANDB_PROJECT: str = "time-series-foundation-model"

    # HuggingFace Hub repository name for model export
    HF_REPO_NAME: str = "patchtst-foundation-pretrained"

    # Local directory for checkpoint storage
    CHECKPOINT_DIR: str = "checkpoints"

    # Google Drive directory for checkpoint storage (Colab)
    GDRIVE_CHECKPOINT_DIR: str = "/content/drive/MyDrive/checkpoints"
