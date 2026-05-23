"""Fine-tune evaluation module for the PatchTST model on ETTh1.

This module implements the full fine-tuning evaluation pipeline: it loads a
pretrained PatchTST backbone, unfreezes ALL encoder layers, attaches a
ProbabilisticForecastHead, and trains the entire model on the ETTh1 train split
for 10 epochs with AdamW(lr=5e-5). After training, it evaluates on the test
split computing MAE, MSE, MASE, and CRPS metrics with inference timing.

This differs from forecasting/finetune.py which freezes the encoder and only
trains the head. Here, all parameters are trainable to establish the upper
bound of performance.

Related modules:
    - model/patchtst.py provides the PatchTSTModel encoder backbone.
    - forecasting/probabilistic_head.py provides ProbabilisticForecastHead
      and quantile_loss for training.
    - forecasting/inference.py provides zero_shot_forecast and compute_num_windows
      for generating test predictions after fine-tuning.
    - data/dataset.py provides TimeSeriesDataset for creating training batches.
    - data/preprocess.py provides inverse_normalize for metric computation.
    - evaluation/metrics.py provides mae, mse, mase, crps_quantile.
    - config.py supplies architecture and forecasting hyperparameters.
"""

import math
import os
import time
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TimeSeriesDataset
from data.preprocess import inverse_normalize
from evaluation.metrics import crps_quantile, mae, mase, mse
from forecasting.inference import compute_num_windows, zero_shot_forecast
from forecasting.probabilistic_head import ProbabilisticForecastHead, quantile_loss
from model.patchtst import PatchTSTModel


def run_finetune_evaluation(
    pretrained_checkpoint_path: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    device: str = "cpu",
    save_dir: str = "checkpoints",
) -> dict[str, Any]:
    """Fine-tune pretrained backbone on ETTh1 and evaluate.

    Steps:
        1. Load pretrained checkpoint, unfreeze ALL encoder layers
        2. Attach ProbabilisticForecastHead
        3. Train with AdamW(lr=5e-5), batch_size=32, 10 epochs
        4. Halt on NaN loss (report epoch number)
        5. Save fine-tuned checkpoint
        6. Evaluate on test split: MAE, MSE, MASE, CRPS
        7. Measure inference time on test set

    Args:
        pretrained_checkpoint_path: Path to the pretrained model checkpoint (.pt).
        train_data: 1D numpy array of normalized ETTh1 training split.
        val_data: 1D numpy array of normalized ETTh1 validation split.
        test_data: 1D numpy array of normalized ETTh1 test split.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        device: Device string ("cpu" or "cuda").
        save_dir: Directory to save the fine-tuned checkpoint.

    Returns:
        {"metrics": {"mae", "mse", "mase", "crps", "inference_time"},
         "train_losses": list[float], "val_losses": list[float],
         "epochs_completed": int, "checkpoint_path": str}

    Raises:
        FileNotFoundError: If pretrained checkpoint does not exist.
        RuntimeError: If NaN loss detected (includes epoch number).
    """
    # -----------------------------------------------------------------------
    # Step 0: Validate checkpoint exists
    # -----------------------------------------------------------------------
    if not os.path.isfile(pretrained_checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{pretrained_checkpoint_path}'. "
            f"Cannot proceed with fine-tuning."
        )

    # -----------------------------------------------------------------------
    # Step 1: Load pretrained checkpoint and unfreeze ALL encoder layers
    # -----------------------------------------------------------------------
    torch_device = torch.device(device)

    checkpoint = torch.load(
        pretrained_checkpoint_path, map_location=torch_device, weights_only=False
    )

    # Instantiate model and head
    model = PatchTSTModel(Config)
    head = ProbabilisticForecastHead(
        d_model=Config.D_MODEL,
        num_patches=Config.NUM_PATCHES,
        forecast_horizon=Config.FORECAST_HORIZON,
        quantiles=Config.QUANTILES,
    )

    # Load pretrained encoder weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load head weights if available (otherwise random init)
    if "head_state_dict" in checkpoint:
        head.load_state_dict(checkpoint["head_state_dict"])

    # Unfreeze ALL encoder layers — all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    # Head parameters are also trainable
    for param in head.parameters():
        param.requires_grad = True

    # Move to device
    model = model.to(torch_device)
    head = head.to(torch_device)

    # -----------------------------------------------------------------------
    # Step 2: Create data loaders
    # -----------------------------------------------------------------------
    train_dataset = TimeSeriesDataset(
        data=train_data,
        context_length=Config.CONTEXT_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
    )

    val_dataset = TimeSeriesDataset(
        data=val_data,
        context_length=Config.CONTEXT_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
    )

    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # -----------------------------------------------------------------------
    # Step 3: Set up optimizer — AdamW with lr=5e-5, all parameters
    # -----------------------------------------------------------------------
    all_params = list(model.parameters()) + list(head.parameters())
    optimizer = AdamW(all_params, lr=5e-5)

    # -----------------------------------------------------------------------
    # Step 4: Training loop — 10 epochs
    # -----------------------------------------------------------------------
    num_epochs = 10
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        head.train()

        epoch_total_loss = 0.0
        epoch_num_batches = 0

        for context_batch, target_batch in train_loader:
            context_batch = context_batch.to(torch_device)
            target_batch = target_batch.to(torch_device)

            # Forward pass through encoder and head
            encoder_output = model(context_batch)
            predictions = head(encoder_output)

            # Compute quantile loss
            loss = quantile_loss(predictions, target_batch, Config.QUANTILES)

            # Check for NaN loss — halt immediately with RuntimeError
            if math.isnan(loss.item()):
                raise RuntimeError(
                    f"NaN loss detected during training at epoch {epoch + 1}. "
                    f"Training halted."
                )

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_num_batches += 1

        # Compute average training loss for this epoch
        avg_train_loss = (
            epoch_total_loss / epoch_num_batches if epoch_num_batches > 0 else 0.0
        )
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        head.eval()

        val_total_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
            for context_batch, target_batch in val_loader:
                context_batch = context_batch.to(torch_device)
                target_batch = target_batch.to(torch_device)

                encoder_output = model(context_batch)
                predictions = head(encoder_output)
                loss = quantile_loss(predictions, target_batch, Config.QUANTILES)

                val_total_loss += loss.item()
                val_num_batches += 1

        avg_val_loss = (
            val_total_loss / val_num_batches if val_num_batches > 0 else 0.0
        )
        val_losses.append(avg_val_loss)

    # -----------------------------------------------------------------------
    # Step 5: Save fine-tuned checkpoint
    # -----------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "finetuned_patchtst_full.pt")

    save_dict = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "epochs_completed": num_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(save_dict, checkpoint_path)

    # -----------------------------------------------------------------------
    # Step 6: Evaluate on test split — MAE, MSE, MASE, CRPS
    # -----------------------------------------------------------------------
    model.eval()
    head.eval()

    # Measure inference time on test set
    inference_start = time.time()

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

    inference_time = time.time() - inference_start

    # Extract actual target values for each window in original scale
    num_windows = forecasts.shape[0]
    actuals_list: list[np.ndarray] = []

    for window_idx in range(num_windows):
        target_start = window_idx * Config.FORECAST_HORIZON + Config.CONTEXT_LENGTH
        target_end = target_start + Config.FORECAST_HORIZON
        actual_normalized = test_data[target_start:target_end]
        actual_original = inverse_normalize(actual_normalized, norm_stats)
        actuals_list.append(actual_original)

    actuals = np.stack(actuals_list, axis=0)

    # P50 (median) as point forecast — index 1 in quantile dim
    p50_forecasts = forecasts[:, :, 1]

    # Compute metrics
    mae_value = mae(p50_forecasts, actuals)
    mse_value = mse(p50_forecasts, actuals)
    mase_value = mase(p50_forecasts, actuals, seasonal_period=24)
    crps_value = crps_quantile(
        q_predictions=forecasts,
        targets=actuals,
        quantiles=Config.QUANTILES,
    )

    # Round inference time to 2 decimal places
    inference_time = round(inference_time, 2)

    # -----------------------------------------------------------------------
    # Step 7: Return results
    # -----------------------------------------------------------------------
    return {
        "metrics": {
            "mae": mae_value,
            "mse": mse_value,
            "mase": mase_value,
            "crps": crps_value,
            "inference_time": inference_time,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_completed": num_epochs,
        "checkpoint_path": checkpoint_path,
    }
