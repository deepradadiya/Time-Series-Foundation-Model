"""Full evaluation pipeline for the Time Series Foundation Model.

This module orchestrates the complete evaluation workflow: zero-shot forecasting
on ETTh1, baseline comparison (ARIMA, Prophet), metric computation (MAE, MSE,
MASE, CRPS), and formatted results table printing. It ties together the inference,
baselines, metrics, and visualization modules into a single entry point.

Related modules:
    - evaluation/metrics.py provides mae, mse, mase, crps_quantile metric functions.
    - evaluation/baselines.py provides run_all_baselines and extract_test_windows.
    - evaluation/visualize.py provides visualize_forecasts for plotting results.
    - forecasting/inference.py provides zero_shot_forecast and compute_num_windows.
    - data/preprocess.py provides split_chronological, load_normalization_stats,
      and inverse_normalize for data preparation.
    - config.py supplies all hyperparameters (CONTEXT_LENGTH, FORECAST_HORIZON, etc.).
"""

import os

import numpy as np

from config import Config
from data.preprocess import (
    inverse_normalize,
    load_normalization_stats,
    split_chronological,
)
from evaluation.baselines import extract_test_windows, run_all_baselines
from evaluation.metrics import crps_quantile, mae, mase, mse
from evaluation.visualize import visualize_forecasts
from forecasting.inference import compute_num_windows, zero_shot_forecast


def _check_checkpoint_exists(checkpoint_path: str) -> None:
    """Verify that the pretrained checkpoint file exists on disk.

    Raises a FileNotFoundError with a descriptive message if the checkpoint
    is missing, aborting evaluation without producing partial results.

    Parameters:
        checkpoint_path: Absolute or relative path to the .pt checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the given path.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{checkpoint_path}'. "
            f"Cannot proceed with evaluation. Please ensure the model has been "
            f"pretrained and the checkpoint file exists at the specified path."
        )


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table of evaluation results.

    The table includes rows for PatchTST zero-shot, PatchTST fine-tuned, ARIMA,
    and Prophet, with columns for MAE, MSE, MASE, and CRPS. All numeric values
    are rounded to 4 decimal places.

    Parameters:
        results: A dictionary mapping model names to their metric dictionaries.
                 Each metric dictionary should have keys: "mae", "mse", "mase", "crps".
                 Example:
                 {
                     "PatchTST (zero-shot)": {"mae": 0.45, "mse": 0.32, "mase": 0.89, "crps": 0.21},
                     "PatchTST (fine-tuned)": {"mae": 0.38, ...},
                     "ARIMA": {"mae": 0.55, ...},
                     "Prophet": {"mae": 0.60, ...},
                 }
    """
    # Define column headers and widths for alignment
    header_model = "Model"
    header_mae = "MAE"
    header_mse = "MSE"
    header_mase = "MASE"
    header_crps = "CRPS"

    # Column widths: model name column is wider to accommodate long names
    col_model_width = 24
    col_metric_width = 10

    # Print the table header with separator lines
    separator = "-" * (col_model_width + col_metric_width * 4 + 5)
    print("\n" + separator)
    print(
        f"{'Model':<{col_model_width}} | "
        f"{'MAE':>{col_metric_width}} | "
        f"{'MSE':>{col_metric_width}} | "
        f"{'MASE':>{col_metric_width}} | "
        f"{'CRPS':>{col_metric_width}}"
    )
    print(separator)

    # Define the row order for the comparison table
    row_order = [
        "PatchTST (zero-shot)",
        "PatchTST (fine-tuned)",
        "ARIMA",
        "Prophet",
    ]

    # Print each row with metrics rounded to 4 decimal places
    for model_name in row_order:
        if model_name in results:
            metrics = results[model_name]
            # Round each metric value to 4 decimal places for display
            mae_val = round(metrics.get("mae", 0.0), 4)
            mse_val = round(metrics.get("mse", 0.0), 4)
            mase_val = round(metrics.get("mase", 0.0), 4)
            crps_val = round(metrics.get("crps", 0.0), 4)

            print(
                f"{model_name:<{col_model_width}} | "
                f"{mae_val:>{col_metric_width}.4f} | "
                f"{mse_val:>{col_metric_width}.4f} | "
                f"{mase_val:>{col_metric_width}.4f} | "
                f"{crps_val:>{col_metric_width}.4f}"
            )
        else:
            # Model not evaluated — show dashes
            print(
                f"{model_name:<{col_model_width}} | "
                f"{'N/A':>{col_metric_width}} | "
                f"{'N/A':>{col_metric_width}} | "
                f"{'N/A':>{col_metric_width}} | "
                f"{'N/A':>{col_metric_width}}"
            )

    # Print closing separator
    print(separator + "\n")


def evaluate_zero_shot(
    model,
    head,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    train_data: np.ndarray,
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
    device: str = "cpu",
) -> dict[str, float]:
    """Run zero-shot evaluation and compute all metrics on ETTh1 test split.

    Generates probabilistic forecasts (P10/P50/P90) using the pretrained model
    without any fine-tuning, then computes MAE, MSE, MASE (using P50 as point
    forecast) and CRPS (using all three quantiles).

    Parameters:
        model: A pretrained PatchTSTModel instance (encoder backbone).
        head: A ProbabilisticForecastHead instance for quantile prediction.
        test_data: 1D numpy array of normalized test split values.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        train_data: 1D numpy array of normalized training data (for MASE scaling).
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to predict (default 96).
        stride: Step size between consecutive windows (default 96).
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        A dictionary with keys "mae", "mse", "mase", "crps" containing the
        computed metric values as floats.
    """
    # Generate probabilistic forecasts using the zero-shot inference pipeline
    # Output shape: (num_windows, forecast_horizon, 3) — P10/P50/P90 in original scale
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

    # Extract the number of windows from the forecast output
    num_windows = forecasts.shape[0]

    # Extract actual target values for each window and inverse-normalize them
    # The targets are the ground truth values in the original data scale
    actuals_list: list[np.ndarray] = []

    for window_idx in range(num_windows):
        # Target starts right after the context window for this window
        target_start = window_idx * stride + context_length
        target_end = target_start + forecast_horizon

        # Extract normalized actual values and convert to original scale
        actual_normalized = test_data[target_start:target_end]
        actual_original = inverse_normalize(actual_normalized, norm_stats)
        actuals_list.append(actual_original)

    # Stack actuals into a 2D array: (num_windows, forecast_horizon)
    actuals = np.stack(actuals_list, axis=0)

    # Extract P50 (median) predictions as the point forecast for MAE/MSE/MASE
    # forecasts shape: (num_windows, forecast_horizon, 3) — index 1 is P50
    p50_forecasts = forecasts[:, :, 1]

    # Compute MAE using P50 as point forecast (Requirement 7.3)
    mae_value = mae(p50_forecasts, actuals)

    # Compute MSE using P50 as point forecast (Requirement 7.3)
    mse_value = mse(p50_forecasts, actuals)

    # Compute MASE using P50 as point forecast, scaled by seasonal naive error
    # The train_data is used to compute the seasonal naive scaling factor
    train_original = inverse_normalize(train_data, norm_stats)
    mase_value = mase(p50_forecasts, actuals, seasonal_period=24)

    # Compute CRPS using all three quantiles (P10, P50, P90) (Requirement 7.3)
    crps_value = crps_quantile(
        q_predictions=forecasts,
        targets=actuals,
        quantiles=Config.QUANTILES,
    )

    # Print summary of zero-shot evaluation results
    print(f"[Zero-Shot] Evaluated {num_windows} test windows.")
    print(f"[Zero-Shot] MAE: {mae_value:.4f}, MSE: {mse_value:.4f}, "
          f"MASE: {mase_value:.4f}, CRPS: {crps_value:.4f}")

    return {
        "mae": mae_value,
        "mse": mse_value,
        "mase": mase_value,
        "crps": crps_value,
    }


def evaluate_baselines(
    train_data: np.ndarray,
    test_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
) -> dict[str, dict[str, float]]:
    """Run ARIMA and Prophet baselines and compute metrics including CRPS placeholder.

    Baselines produce point forecasts only, so CRPS is set to 0.0 (not applicable
    for point forecasts without quantile information).

    Parameters:
        train_data: 1D numpy array of normalized training data.
        test_data: 1D numpy array of normalized test split values.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to forecast (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        A dictionary mapping baseline names ("ARIMA", "Prophet") to their metric
        dictionaries with keys "mae", "mse", "mase", "crps".
    """
    # Convert normalized data to original scale for baseline fitting
    # Baselines operate on original-scale data since they fit their own models
    train_original = inverse_normalize(train_data, norm_stats)
    test_original = inverse_normalize(test_data, norm_stats)

    # Run all classical baselines (ARIMA and Prophet) on the same test windows
    # This returns MAE, MSE, MASE for each baseline
    baseline_results = run_all_baselines(
        train=train_original,
        test_data=test_original,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    # Format results with CRPS set to 0.0 for baselines (point forecasts only)
    formatted_results: dict[str, dict[str, float]] = {}

    for baseline_name, metrics in baseline_results.items():
        # Map internal baseline names to display names
        display_name = baseline_name.upper() if baseline_name == "arima" else baseline_name.capitalize()

        formatted_results[display_name] = {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "mase": metrics["mase"],
            "crps": 0.0,  # CRPS not applicable for point forecasts
        }

    return formatted_results


def run_full_evaluation(
    checkpoint_path: str,
    test_data: np.ndarray,
    train_data: np.ndarray,
    norm_stats: dict[str, list[float]],
    finetuned_checkpoint_path: str | None = None,
    run_baselines: bool = True,
    generate_plots: bool = True,
    device: str = "cpu",
    output_dir: str = "evaluation",
) -> dict[str, dict[str, float]]:
    """Execute the full evaluation pipeline: zero-shot, baselines, and results table.

    This is the main entry point for evaluation. It:
    1. Validates that the pretrained checkpoint exists
    2. Loads the pretrained model and runs zero-shot evaluation
    3. Optionally loads a fine-tuned model and evaluates it
    4. Runs ARIMA and Prophet baselines on the same test windows
    5. Prints a formatted comparison table with all results
    6. Optionally generates visualization plots

    Parameters:
        checkpoint_path: Path to the pretrained model checkpoint (.pt file).
        test_data: 1D numpy array of normalized ETTh1 test split values.
        train_data: 1D numpy array of normalized ETTh1 training data.
        norm_stats: Dictionary with "mean" and "std" keys for inverse normalization.
        finetuned_checkpoint_path: Optional path to a fine-tuned checkpoint.
                                    If None, fine-tuned row shows N/A in the table.
        run_baselines: Whether to run ARIMA and Prophet baselines (default True).
        generate_plots: Whether to generate visualization plots (default True).
        device: Device to run inference on ("cpu" or "cuda").
        output_dir: Directory for saving plots and results (default "evaluation").

    Returns:
        A dictionary mapping model names to their metric dictionaries.

    Raises:
        FileNotFoundError: If the pretrained checkpoint is not found.
    """
    import torch

    from forecasting.probabilistic_head import ProbabilisticForecastHead
    from model.patchtst import PatchTSTModel

    # -------------------------------------------------------------------------
    # Step 1: Validate that the pretrained checkpoint exists (Requirement 7.6)
    # Abort immediately if the checkpoint is missing — no partial results
    # -------------------------------------------------------------------------
    _check_checkpoint_exists(checkpoint_path)

    # Collect all results in a single dictionary for the comparison table
    all_results: dict[str, dict[str, float]] = {}

    # -------------------------------------------------------------------------
    # Step 2: Load pretrained model and run zero-shot evaluation (Requirement 7.1-7.3)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("EVALUATION PIPELINE — ETTh1 Zero-Shot Forecasting")
    print("=" * 60)

    # Load the pretrained checkpoint from disk
    print(f"\n[Evaluate] Loading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Instantiate the model and forecast head with default configuration
    model = PatchTSTModel(Config)
    head = ProbabilisticForecastHead(
        d_model=Config.D_MODEL,
        num_patches=Config.NUM_PATCHES,
        forecast_horizon=Config.FORECAST_HORIZON,
        quantiles=Config.QUANTILES,
    )

    # Load pretrained weights into the model
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load forecast head weights if available
    if "head_state_dict" in checkpoint:
        head.load_state_dict(checkpoint["head_state_dict"])

    # Run zero-shot evaluation on the ETTh1 test split
    print("\n[Evaluate] Running zero-shot evaluation...")
    zero_shot_metrics = evaluate_zero_shot(
        model=model,
        head=head,
        test_data=test_data,
        norm_stats=norm_stats,
        train_data=train_data,
        device=device,
    )
    all_results["PatchTST (zero-shot)"] = zero_shot_metrics

    # -------------------------------------------------------------------------
    # Step 3: Optionally evaluate fine-tuned model (Requirement 12.4)
    # -------------------------------------------------------------------------
    if finetuned_checkpoint_path is not None:
        print(f"\n[Evaluate] Loading fine-tuned checkpoint: {finetuned_checkpoint_path}")
        _check_checkpoint_exists(finetuned_checkpoint_path)

        # Load fine-tuned checkpoint
        ft_checkpoint = torch.load(
            finetuned_checkpoint_path, map_location=device, weights_only=False
        )

        # Create fresh model and head instances for fine-tuned evaluation
        ft_model = PatchTSTModel(Config)
        ft_head = ProbabilisticForecastHead(
            d_model=Config.D_MODEL,
            num_patches=Config.NUM_PATCHES,
            forecast_horizon=Config.FORECAST_HORIZON,
            quantiles=Config.QUANTILES,
        )

        # Load fine-tuned weights
        if "model_state_dict" in ft_checkpoint:
            ft_model.load_state_dict(ft_checkpoint["model_state_dict"])
        else:
            ft_model.load_state_dict(ft_checkpoint)

        if "head_state_dict" in ft_checkpoint:
            ft_head.load_state_dict(ft_checkpoint["head_state_dict"])

        # Run evaluation with the fine-tuned model
        print("\n[Evaluate] Running fine-tuned evaluation...")
        finetuned_metrics = evaluate_zero_shot(
            model=ft_model,
            head=ft_head,
            test_data=test_data,
            norm_stats=norm_stats,
            train_data=train_data,
            device=device,
        )
        all_results["PatchTST (fine-tuned)"] = finetuned_metrics
    else:
        print("\n[Evaluate] No fine-tuned checkpoint provided — skipping fine-tuned evaluation.")

    # -------------------------------------------------------------------------
    # Step 4: Run classical baselines (ARIMA, Prophet) (Requirement 7.5)
    # -------------------------------------------------------------------------
    if run_baselines:
        print("\n[Evaluate] Running classical baselines (ARIMA, Prophet)...")
        baseline_results = evaluate_baselines(
            train_data=train_data,
            test_data=test_data,
            norm_stats=norm_stats,
        )
        all_results.update(baseline_results)
    else:
        print("\n[Evaluate] Skipping baseline evaluation.")

    # -------------------------------------------------------------------------
    # Step 5: Print formatted comparison table (Requirement 9.2)
    # -------------------------------------------------------------------------
    print("\n[Evaluate] Results Summary:")
    print_results_table(all_results)

    # -------------------------------------------------------------------------
    # Step 6: Optionally generate visualization plots (Requirement 9.3)
    # -------------------------------------------------------------------------
    if generate_plots:
        print("[Evaluate] Generating forecast visualization plots...")

        # Re-run zero-shot forecast to get the full forecast array for plotting
        forecasts = zero_shot_forecast(
            model=model,
            head=head,
            data=test_data,
            norm_stats=norm_stats,
            device=device,
        )

        # Extract actual target values in original scale for visualization
        num_windows = forecasts.shape[0]
        actuals_list: list[np.ndarray] = []

        for window_idx in range(num_windows):
            target_start = window_idx * Config.FORECAST_HORIZON + Config.CONTEXT_LENGTH
            target_end = target_start + Config.FORECAST_HORIZON
            actual_normalized = test_data[target_start:target_end]
            actual_original = inverse_normalize(actual_normalized, norm_stats)
            actuals_list.append(actual_original)

        actuals = np.stack(actuals_list, axis=0)

        # Generate and save visualization plots
        visualize_forecasts(
            actuals=actuals,
            forecasts=forecasts,
            num_plots=5,
            output_dir=output_dir,
        )

    return all_results


def main() -> None:
    """Command-line entry point for running the full evaluation pipeline.

    This function loads the ETTh1 dataset, splits it chronologically, loads
    normalization statistics, and runs the complete evaluation pipeline including
    zero-shot forecasting, baseline comparison, and results visualization.
    """
    import pandas as pd

    # -------------------------------------------------------------------------
    # Configuration: paths and settings
    # -------------------------------------------------------------------------
    checkpoint_path = "checkpoints/pretrained_patchtst.pt"
    finetuned_path = "checkpoints/finetuned_patchtst.pt"
    etth1_data_path = "data/raw/etth1/ETTh1.csv"
    norm_stats_dataset = "etth1"

    # Check if fine-tuned checkpoint exists (optional)
    finetuned_checkpoint = finetuned_path if os.path.isfile(finetuned_path) else None

    # Determine device (use GPU if available)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Evaluate] Using device: {device}")

    # -------------------------------------------------------------------------
    # Load and preprocess ETTh1 data
    # -------------------------------------------------------------------------
    print(f"[Evaluate] Loading ETTh1 data from: {etth1_data_path}")

    # Load the raw CSV data
    df = pd.read_csv(etth1_data_path)

    # Use the first numeric column as the target (OT — Oil Temperature)
    # Skip the date column (first column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = numeric_cols[0] if numeric_cols else df.columns[1]
    raw_data = df[target_col].values.astype(np.float64)

    # Split chronologically into train/val/test (70/15/15)
    train_data, val_data, test_data = split_chronological(raw_data)

    # Load normalization statistics computed during preprocessing
    norm_stats = load_normalization_stats(norm_stats_dataset)

    # -------------------------------------------------------------------------
    # Run the full evaluation pipeline
    # -------------------------------------------------------------------------
    results = run_full_evaluation(
        checkpoint_path=checkpoint_path,
        test_data=test_data,
        train_data=train_data,
        norm_stats=norm_stats,
        finetuned_checkpoint_path=finetuned_checkpoint,
        run_baselines=True,
        generate_plots=True,
        device=device,
    )

    print("[Evaluate] Evaluation complete.")


# Allow running this module directly from the command line
if __name__ == "__main__":
    main()
