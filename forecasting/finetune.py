"""Fine-tuning module for the PatchTST model on the ETTh1 dataset.

This module implements the fine-tuning pipeline that loads pretrained PatchTST
encoder weights, freezes the encoder parameters, and trains only the
ProbabilisticForecastHead on the ETTh1 training split. It demonstrates the
upper bound of performance achievable with task-specific adaptation and
provides results for the comparison table alongside zero-shot and baselines.

Related modules:
    - model/patchtst.py provides the PatchTSTModel encoder backbone.
    - forecasting/probabilistic_head.py provides ProbabilisticForecastHead
      and quantile_loss for training the forecast head.
    - forecasting/inference.py provides zero_shot_forecast and compute_num_windows
      for generating test predictions after fine-tuning.
    - data/dataset.py provides TimeSeriesDataset for creating training batches.
    - data/preprocess.py provides split_chronological, load_normalization_stats,
      and inverse_normalize for data preparation.
    - evaluation/metrics.py provides mae, mse, crps_quantile for test evaluation.
    - evaluation/evaluate.py provides print_results_table for displaying results.
    - config.py supplies FINETUNE_LR (1e-5), FINETUNE_EPOCHS (10),
      FINETUNE_BATCH_SIZE (32), and other hyperparameters.
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TimeSeriesDataset
from evaluation.metrics import crps_quantile, mae, mse
from forecasting.inference import compute_num_windows, zero_shot_forecast
from forecasting.probabilistic_head import ProbabilisticForecastHead, quantile_loss
from model.patchtst import PatchTSTModel


def _create_finetune_lr_scheduler(
    optimizer: AdamW,
    total_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """Create a cosine learning rate scheduler with linear warmup for fine-tuning.

    The schedule has two phases:
    1. Linear warmup: LR increases linearly from 0 to base_lr over warmup_epochs.
    2. Cosine decay: LR decreases from base_lr to min_lr following a cosine curve
       over the remaining epochs.

    Args:
        optimizer: The AdamW optimizer whose learning rate will be scheduled.
        total_epochs: Total number of fine-tuning epochs (10).
        warmup_epochs: Number of epochs for linear warmup (1).
        min_lr: Minimum learning rate at the end of cosine decay (1e-6).
        base_lr: Base (peak) learning rate after warmup (1e-5).

    Returns:
        A LambdaLR scheduler that adjusts the learning rate each epoch.
    """

    def lr_lambda(current_epoch: int) -> float:
        """Compute the learning rate multiplier for a given epoch.

        Args:
            current_epoch: The current epoch number (0-indexed).

        Returns:
            A float multiplier applied to the base learning rate.
        """
        # Phase 1: Linear warmup from 0 to base_lr
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs

        # Phase 2: Cosine decay from base_lr to min_lr
        decay_epochs = total_epochs - warmup_epochs
        progress = (current_epoch - warmup_epochs) / decay_epochs

        # Cosine decay formula
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale so that the minimum multiplier corresponds to min_lr
        min_multiplier = min_lr / base_lr
        return min_multiplier + (1.0 - min_multiplier) * cosine_decay

    # Create and return the LambdaLR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def _compute_val_loss(
    model: PatchTSTModel,
    head: ProbabilisticForecastHead,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average validation loss on the ETTh1 validation split.

    Runs the model and head in evaluation mode (no dropout, no gradients)
    and returns the mean quantile loss across all validation batches.

    Args:
        model: The PatchTST encoder (frozen during fine-tuning).
        head: The ProbabilisticForecastHead being trained.
        val_loader: DataLoader wrapping the validation TimeSeriesDataset.
        device: The device (CPU or GPU) to run computations on.

    Returns:
        The average validation quantile loss as a float. Returns 0.0 if empty.
    """
    # Set both model and head to evaluation mode (disables dropout)
    model.eval()
    head.eval()

    total_loss = 0.0
    num_batches = 0

    # Disable gradient computation for validation efficiency
    with torch.no_grad():
        for context_batch, target_batch in val_loader:
            # Move data to the appropriate device
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass through frozen encoder
            encoder_output = model(context_batch)

            # Forward pass through the forecast head
            predictions = head(encoder_output)

            # Compute quantile loss (pinball loss)
            loss = quantile_loss(predictions, target_batch, Config.QUANTILES)

            total_loss += loss.item()
            num_batches += 1

    # Return average loss, or 0.0 if no batches
    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def _evaluate_on_test(
    model: PatchTSTModel,
    head: ProbabilisticForecastHead,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate the fine-tuned model on the ETTh1 test split.

    Generates probabilistic forecasts using the same sliding window approach
    as zero-shot evaluation, then computes MAE, MSE, and CRPS metrics.

    Args:
        model: The PatchTST encoder model.
        head: The fine-tuned ProbabilisticForecastHead.
        test_data: 1D numpy array of normalized ETTh1 test split values.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        device: Device string ("cpu" or "cuda").

    Returns:
        A dictionary with keys "mae", "mse", "crps" containing metric values.
    """
    # Generate forecasts using the same pipeline as zero-shot inference
    # Output shape: (num_windows, forecast_horizon, 3) in original scale
    forecasts = zero_shot_forecast(
        model=model,
        head=head,
        data=test_data,
        norm_stats=norm_stats,
        context_length=Config.CONTEXT_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        stride=Config.FORECAST_HORIZON,
        device=device,
    )

    num_windows = forecasts.shape[0]

    # Extract actual target values for each window in original scale
    from data.preprocess import inverse_normalize

    actuals_list: list[np.ndarray] = []
    for window_idx in range(num_windows):
        target_start = window_idx * Config.FORECAST_HORIZON + Config.CONTEXT_LENGTH
        target_end = target_start + Config.FORECAST_HORIZON
        actual_normalized = test_data[target_start:target_end]
        actual_original = inverse_normalize(actual_normalized, norm_stats)
        actuals_list.append(actual_original)

    # Stack actuals: (num_windows, forecast_horizon)
    actuals = np.stack(actuals_list, axis=0)

    # Extract P50 (median) as point forecast — index 1 in the quantile dim
    p50_forecasts = forecasts[:, :, 1]

    # Compute metrics
    mae_value = mae(p50_forecasts, actuals)
    mse_value = mse(p50_forecasts, actuals)
    crps_value = crps_quantile(
        q_predictions=forecasts,
        targets=actuals,
        quantiles=Config.QUANTILES,
    )

    return {"mae": mae_value, "mse": mse_value, "crps": crps_value}


def finetune(
    pretrained_checkpoint_path: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    config: type = Config,
    device: Optional[torch.device] = None,
    save_dir: str = "checkpoints",
) -> dict[str, object]:
    """Fine-tune the ProbabilisticForecastHead on ETTh1 with frozen encoder.

    This function implements the complete fine-tuning pipeline:
    1. Load pretrained PatchTST encoder weights from checkpoint
    2. Freeze all encoder parameters (no gradient updates)
    3. Train only the ProbabilisticForecastHead for 10 epochs
    4. Use cosine LR schedule with warmup over the first epoch
    5. Track validation loss and select the best checkpoint
    6. Apply early stopping if NaN loss or 3 consecutive val increases
    7. Evaluate the best model on the test split (MAE, MSE, CRPS)

    Args:
        pretrained_checkpoint_path: Path to the pretrained model checkpoint (.pt).
        train_data: 1D numpy array of normalized ETTh1 training split.
        val_data: 1D numpy array of normalized ETTh1 validation split.
        test_data: 1D numpy array of normalized ETTh1 test split.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        config: Configuration class with fine-tuning hyperparameters.
        device: Torch device to train on. If None, auto-detects GPU/CPU.
        save_dir: Directory to save the fine-tuned checkpoint.

    Returns:
        A dictionary containing:
        {
            "test_metrics": {"mae": float, "mse": float, "crps": float},
            "best_val_loss": float,
            "epochs_completed": int,
            "stopped_early": bool,
            "checkpoint_path": str (path to saved best model),
        }

    Raises:
        FileNotFoundError: If the pretrained checkpoint does not exist.
    """
    # -----------------------------------------------------------------------
    # Device setup: prefer GPU if available, fall back to CPU
    # -----------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    print(f"[Finetune] Using device: {device}")

    # -----------------------------------------------------------------------
    # Step 1: Load pretrained model weights (Requirement 12.1)
    # -----------------------------------------------------------------------
    if not os.path.isfile(pretrained_checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{pretrained_checkpoint_path}'. "
            f"Cannot proceed with fine-tuning. Please pretrain the model first."
        )

    print(f"[Finetune] Loading pretrained checkpoint: {pretrained_checkpoint_path}")
    checkpoint = torch.load(
        pretrained_checkpoint_path, map_location=device, weights_only=False
    )

    # Instantiate the model and forecast head
    model = PatchTSTModel(config)
    head = ProbabilisticForecastHead(
        d_model=config.D_MODEL,
        num_patches=config.NUM_PATCHES,
        forecast_horizon=config.FORECAST_HORIZON,
        quantiles=config.QUANTILES,
    )

    # Load pretrained encoder weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load head weights if available (otherwise start from random init)
    if "head_state_dict" in checkpoint:
        head.load_state_dict(checkpoint["head_state_dict"])

    # -----------------------------------------------------------------------
    # Step 2: Freeze encoder parameters (Requirement 12.1)
    # Only the ProbabilisticForecastHead will be trained.
    # -----------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # Move model and head to the target device
    model = model.to(device)
    head = head.to(device)

    # Set encoder to eval mode permanently (no dropout during fine-tuning)
    model.eval()

    # Count trainable parameters (should only be the head)
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[Finetune] Trainable parameters (head only): {trainable_params:,}")

    # -----------------------------------------------------------------------
    # Step 3: Create data loaders for training and validation
    # -----------------------------------------------------------------------
    train_dataset = TimeSeriesDataset(
        data=train_data,
        context_length=config.CONTEXT_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
    )

    val_dataset = TimeSeriesDataset(
        data=val_data,
        context_length=config.CONTEXT_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.FINETUNE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(f"[Finetune] Training samples: {len(train_dataset)}, "
          f"Validation samples: {len(val_dataset)}")

    # -----------------------------------------------------------------------
    # Step 4: Set up optimizer and LR scheduler (Requirement 12.2)
    # Only head parameters are optimized; lr=1e-5, cosine with 1 epoch warmup
    # -----------------------------------------------------------------------
    optimizer = AdamW(
        head.parameters(),
        lr=config.FINETUNE_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine schedule with linear warmup over the first epoch
    scheduler = _create_finetune_lr_scheduler(
        optimizer=optimizer,
        total_epochs=config.FINETUNE_EPOCHS,
        warmup_epochs=1,
        min_lr=config.MIN_LR,
        base_lr=config.FINETUNE_LR,
    )

    # -----------------------------------------------------------------------
    # Step 5: Training loop with early stopping (Requirements 12.1-12.5)
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_head_state = None
    best_epoch = 0
    consecutive_increases = 0
    stopped_early = False

    # Store training history
    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"[Finetune] Starting fine-tuning for {config.FINETUNE_EPOCHS} epochs")
    print(f"[Finetune] LR: {config.FINETUNE_LR}, Batch size: {config.FINETUNE_BATCH_SIZE}")

    for epoch in range(config.FINETUNE_EPOCHS):
        # Set head to training mode (enables dropout if any)
        head.train()

        epoch_total_loss = 0.0
        epoch_num_batches = 0

        # -------------------------------------------------------------------
        # Training loop over all batches in the ETTh1 training split
        # -------------------------------------------------------------------
        for context_batch, target_batch in train_loader:
            # Move data to device
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass through frozen encoder (no grad needed for encoder)
            with torch.no_grad():
                encoder_output = model(context_batch)

            # Forward pass through the trainable forecast head
            predictions = head(encoder_output)

            # Compute quantile loss (pinball loss)
            loss = quantile_loss(predictions, target_batch, config.QUANTILES)

            # Check for NaN loss — trigger early stopping (Requirement 12.5)
            if math.isnan(loss.item()):
                print(f"\n[Finetune] ERROR: NaN loss detected at epoch {epoch + 1}. "
                      f"Stopping training early.")
                stopped_early = True
                break

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for epoch average
            epoch_total_loss += loss.item()
            epoch_num_batches += 1

        # If NaN was detected, break out of the epoch loop
        if stopped_early:
            break

        # Compute average training loss for this epoch
        avg_train_loss = (
            epoch_total_loss / epoch_num_batches if epoch_num_batches > 0 else 0.0
        )
        train_losses.append(avg_train_loss)

        # Step the learning rate scheduler
        scheduler.step()

        # -------------------------------------------------------------------
        # Compute validation loss (Requirement 12.3)
        # -------------------------------------------------------------------
        val_loss = _compute_val_loss(model, head, val_loader, device)
        val_losses.append(val_loss)

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Finetune] Epoch {epoch + 1}/{config.FINETUNE_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # -------------------------------------------------------------------
        # Track best model by validation loss (Requirement 12.3)
        # -------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Save a copy of the best head state
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            # Reset consecutive increase counter on improvement
            consecutive_increases = 0
        else:
            # Validation loss did not improve
            consecutive_increases += 1

        # -------------------------------------------------------------------
        # Early stopping: 3 consecutive val loss increases (Requirement 12.5)
        # -------------------------------------------------------------------
        if consecutive_increases >= 3:
            print(
                f"\n[Finetune] Early stopping triggered: validation loss increased "
                f"for {consecutive_increases} consecutive epochs. "
                f"Restoring best checkpoint from epoch {best_epoch}."
            )
            stopped_early = True
            break

    # -----------------------------------------------------------------------
    # Step 6: Restore best checkpoint (Requirement 12.3, 12.5)
    # -----------------------------------------------------------------------
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
        print(f"[Finetune] Restored best model from epoch {best_epoch} "
              f"(val loss: {best_val_loss:.6f})")
    else:
        print("[Finetune] Warning: No valid checkpoint found. Using final state.")

    # -----------------------------------------------------------------------
    # Step 7: Save the fine-tuned model checkpoint
    # -----------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "finetuned_patchtst.pt")

    save_dict = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "config": {
            "finetune_lr": config.FINETUNE_LR,
            "finetune_epochs": config.FINETUNE_EPOCHS,
            "finetune_batch_size": config.FINETUNE_BATCH_SIZE,
        },
    }
    torch.save(save_dict, checkpoint_path)
    print(f"[Finetune] Saved fine-tuned checkpoint: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Step 8: Evaluate on test split (Requirement 12.3)
    # Compute MAE, MSE, CRPS using the same test windows as zero-shot
    # -----------------------------------------------------------------------
    print("\n[Finetune] Evaluating on ETTh1 test split...")
    test_metrics = _evaluate_on_test(
        model=model,
        head=head,
        test_data=test_data,
        norm_stats=norm_stats,
        device=device_str,
    )

    print(f"[Finetune] Test MAE: {test_metrics['mae']:.4f}, "
          f"MSE: {test_metrics['mse']:.4f}, "
          f"CRPS: {test_metrics['crps']:.4f}")

    # -----------------------------------------------------------------------
    # Step 9: Print results for inclusion in comparison table (Requirement 12.4)
    # -----------------------------------------------------------------------
    print("\n[Finetune] Fine-tuned results for comparison table:")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print(f"  CRPS: {test_metrics['crps']:.4f}")

    # Return comprehensive results dictionary
    epochs_completed = len(train_losses)
    return {
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "checkpoint_path": checkpoint_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def main() -> None:
    """Command-line entry point for running the fine-tuning pipeline.

    This function loads the ETTh1 dataset, splits it chronologically, loads
    normalization statistics, and runs the complete fine-tuning pipeline.
    Results are printed as a comparison table row.
    """
    import pandas as pd

    from data.preprocess import (
        load_normalization_stats,
        normalize,
        split_chronological,
        compute_normalization_stats,
    )

    # -------------------------------------------------------------------------
    # Configuration: paths and settings
    # -------------------------------------------------------------------------
    pretrained_path = "checkpoints/pretrained_patchtst.pt"
    etth1_data_path = "data/raw/etth1/ETTh1.csv"
    norm_stats_dataset = "etth1"

    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Finetune] Using device: {device}")

    # -------------------------------------------------------------------------
    # Load and preprocess ETTh1 data
    # -------------------------------------------------------------------------
    print(f"[Finetune] Loading ETTh1 data from: {etth1_data_path}")
    df = pd.read_csv(etth1_data_path)

    # Use the first numeric column as the target (OT — Oil Temperature)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = numeric_cols[0] if numeric_cols else df.columns[1]
    raw_data = df[target_col].values.astype(np.float64)

    # Split chronologically into train/val/test (70/15/15)
    train_raw, val_raw, test_raw = split_chronological(raw_data)

    # Compute normalization statistics from training split only
    norm_stats = compute_normalization_stats(train_raw)

    # Normalize all splits using training statistics
    train_normalized = normalize(train_raw, norm_stats)
    val_normalized = normalize(val_raw, norm_stats)
    test_normalized = normalize(test_raw, norm_stats)

    # -------------------------------------------------------------------------
    # Run fine-tuning pipeline
    # -------------------------------------------------------------------------
    results = finetune(
        pretrained_checkpoint_path=pretrained_path,
        train_data=train_normalized,
        val_data=val_normalized,
        test_data=test_normalized,
        norm_stats=norm_stats,
        device=device,
    )

    print(f"\n[Finetune] Fine-tuning complete!")
    print(f"[Finetune] Epochs completed: {results['epochs_completed']}")
    print(f"[Finetune] Early stopped: {results['stopped_early']}")
    print(f"[Finetune] Best val loss: {results['best_val_loss']:.6f}")
    print(f"[Finetune] Test metrics: {results['test_metrics']}")


# Allow running this module directly from the command line
if __name__ == "__main__":
    main()
