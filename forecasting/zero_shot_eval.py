"""Zero-shot transfer evaluation for the PatchTST foundation model.

This module orchestrates zero-shot evaluation with a frozen pretrained backbone
and a randomly initialized ProbabilisticForecastHead. It loads a pretrained
checkpoint, freezes all encoder parameters, attaches a random forecast head,
runs sliding window inference on the ETTh1 test set, computes metrics
(MAE, MSE, MASE, CRPS), measures inference time, and saves predictions to CSV.

Related modules:
    - model/patchtst.py provides the PatchTSTModel encoder backbone.
    - forecasting/probabilistic_head.py provides the ProbabilisticForecastHead.
    - forecasting/inference.py provides zero_shot_forecast and compute_num_windows.
    - evaluation/metrics.py provides mae, mse, mase, crps_quantile.
    - data/preprocess.py provides inverse_normalize for original-scale actuals.
    - config.py supplies CONTEXT_LENGTH, FORECAST_HORIZON, D_MODEL, NUM_PATCHES, QUANTILES.
"""

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import Config
from data.preprocess import inverse_normalize
from evaluation.metrics import mae, mse, mase, crps_quantile
from forecasting.inference import compute_num_windows, zero_shot_forecast
from forecasting.probabilistic_head import ProbabilisticForecastHead
from model.patchtst import PatchTSTModel


def run_zero_shot_evaluation(
    checkpoint_path: str,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    train_data: np.ndarray,
    context_length: int = 512,
    forecast_horizon: int = 96,
    stride: int = 96,
    device: str = "cpu",
    output_dir: str = "forecasting/results",
) -> dict[str, Any]:
    """Load pretrained backbone, freeze weights, attach random head, evaluate.

    Steps:
        1. Load checkpoint, freeze all encoder params (requires_grad=False)
        2. Attach ProbabilisticForecastHead with random weights (no training)
        3. Run sliding window inference (context=512, horizon=96, stride=96)
        4. Compute MAE, MSE, MASE (P50 as point forecast), CRPS (P10/P50/P90)
        5. Measure inference time
        6. Save predictions to forecasting/results/zero_shot_predictions.csv

    Parameters:
        checkpoint_path: Path to the pretrained model checkpoint (.pt file).
        test_data: 1D numpy array of normalized test set values.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        train_data: 1D numpy array of raw (original scale) training data, used
                    for MASE seasonal naive denominator computation.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to predict (default 96).
        stride: Step size between consecutive windows (default 96).
        device: Device to run inference on ("cpu" or "cuda").
        output_dir: Directory to save prediction CSV (default "forecasting/results").

    Returns:
        {"metrics": {"mae", "mse", "mase", "crps", "inference_time"},
         "forecasts": np.ndarray, "actuals": np.ndarray}

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    # -------------------------------------------------------------------------
    # Step 0: Validate checkpoint exists
    # -------------------------------------------------------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{checkpoint_path}'. "
            f"Cannot perform zero-shot evaluation without a valid checkpoint."
        )

    # -------------------------------------------------------------------------
    # Step 1: Load checkpoint and freeze all encoder parameters
    # -------------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PatchTSTModel(Config)

    # Load pretrained weights into the encoder
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Freeze all encoder parameters — no gradient computation during inference
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2: Attach ProbabilisticForecastHead with random weights
    # The head is NOT loaded from checkpoint — it uses random initialization
    # to demonstrate zero-shot transfer (encoder representations only).
    # -------------------------------------------------------------------------
    head = ProbabilisticForecastHead(
        d_model=Config.D_MODEL,
        num_patches=Config.NUM_PATCHES,
        forecast_horizon=forecast_horizon,
        quantiles=Config.QUANTILES,
    )

    # -------------------------------------------------------------------------
    # Step 3: Run sliding window inference and measure wall-clock time
    # -------------------------------------------------------------------------
    start_time = time.time()

    forecasts = zero_shot_forecast(
        model=model,
        head=head,
        data=test_data,
        norm_stats=norm_stats,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        device=device,
    )

    end_time = time.time()
    inference_time = round(end_time - start_time, 2)

    # forecasts shape: (num_windows, forecast_horizon, 3) — P10, P50, P90

    # -------------------------------------------------------------------------
    # Step 4: Extract actuals in original scale for metric computation
    # -------------------------------------------------------------------------
    num_windows = compute_num_windows(
        data_length=len(test_data),
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    actuals_list = []
    for window_idx in range(num_windows):
        start = window_idx * stride + context_length
        end = start + forecast_horizon
        window_actual_normalized = test_data[start:end]
        # Inverse normalize to get original scale
        window_actual_original = inverse_normalize(window_actual_normalized, norm_stats)
        actuals_list.append(window_actual_original)

    actuals = np.stack(actuals_list, axis=0)  # (num_windows, forecast_horizon)

    # Extract quantile predictions
    p10 = forecasts[:, :, 0]  # (num_windows, forecast_horizon)
    p50 = forecasts[:, :, 1]  # (num_windows, forecast_horizon)
    p90 = forecasts[:, :, 2]  # (num_windows, forecast_horizon)

    # -------------------------------------------------------------------------
    # Step 5: Compute metrics
    # MAE, MSE, MASE use P50 as point forecast
    # CRPS uses all three quantiles (P10, P50, P90)
    # -------------------------------------------------------------------------
    mae_value = mae(p50, actuals)
    mse_value = mse(p50, actuals)
    mase_value = mase(p50, actuals, seasonal_period=24)
    crps_value = crps_quantile(forecasts, actuals, quantiles=Config.QUANTILES)

    metrics = {
        "mae": mae_value,
        "mse": mse_value,
        "mase": mase_value,
        "crps": crps_value,
        "inference_time": inference_time,
    }

    # -------------------------------------------------------------------------
    # Step 6: Save predictions to CSV
    # Columns: window_index, time_step, actual, P10, P50, P90
    # -------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for w_idx in range(num_windows):
        for t_step in range(forecast_horizon):
            rows.append({
                "window_index": w_idx,
                "time_step": t_step,
                "actual": float(actuals[w_idx, t_step]),
                "P10": float(p10[w_idx, t_step]),
                "P50": float(p50[w_idx, t_step]),
                "P90": float(p90[w_idx, t_step]),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "zero_shot_predictions.csv")
    df.to_csv(csv_path, index=False)

    return {
        "metrics": metrics,
        "forecasts": forecasts,
        "actuals": actuals,
    }
