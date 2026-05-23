"""Unified evaluation pipeline entry point for the Time Series Foundation Model.

This script orchestrates the complete evaluation workflow:
    1. Runs Naive, ARIMA, Prophet baselines via forecasting/baselines.py
    2. Runs zero-shot evaluation via forecasting/zero_shot_eval.py
    3. Runs fine-tune evaluation via forecasting/finetune_eval.py
    4. Computes all metrics via evaluation/metrics.py
    5. Prints results table and saves JSON via evaluation/results_table.py
    6. Generates all 3 plots via evaluation/visualize_forecasts.py

All models are evaluated on the ETTh1 test set with consistent windowing:
    - context_length = 512
    - forecast_horizon = 96
    - stride = 96

Usage:
    python run_evaluation.py

Results are saved to:
    - evaluation/results/final_metrics.json (metrics JSON)
    - evaluation/results/forecast_window_0.png (forecast plot)
    - evaluation/results/pretraining_loss_curve.png (loss curve)
    - evaluation/results/mae_comparison_bar_chart.png (MAE bar chart)

Related modules:
    - forecasting/baselines.py: Naive, ARIMA, Prophet baseline runners
    - forecasting/zero_shot_eval.py: Zero-shot transfer evaluation
    - forecasting/finetune_eval.py: Fine-tune evaluation (10 epochs, AdamW lr=5e-5)
    - evaluation/metrics.py: MAE, MSE, MASE, CRPS metric implementations
    - evaluation/results_table.py: Formatted table + JSON export
    - evaluation/visualize_forecasts.py: Publication-quality plots
    - data/preprocess.py: ETTh1 data loading and preprocessing
    - config.py: Central configuration parameters

Requirements validated: 1.3, 1.4, 2.3, 2.4, 3.3, 3.4, 4.6, 4.7, 4.8, 5.4, 5.5, 7.1, 7.3
"""

import os
import time

import numpy as np
import pandas as pd

from config import Config
from data.preprocess import (
    inverse_normalize,
    load_normalization_stats,
    split_chronological,
)
from evaluation.results_table import print_results_table, save_results_json
from evaluation.visualize_forecasts import (
    plot_forecast,
    plot_loss_curve,
    plot_mae_bar_chart,
)
from forecasting.baselines import (
    run_arima_baseline_eval,
    run_naive_baseline,
    run_prophet_baseline_eval,
)
from forecasting.finetune_eval import run_finetune_evaluation
from forecasting.zero_shot_eval import run_zero_shot_evaluation


# ---------------------------------------------------------------------------
# Pipeline configuration — consistent windowing across all models
# ---------------------------------------------------------------------------
CONTEXT_LENGTH = 512
FORECAST_HORIZON = 96
STRIDE = 96

# Paths
ETTH1_DATA_PATH = "data/raw/etth1/ETTh1.csv"
PRETRAINED_CHECKPOINT = "checkpoints/pretrained_patchtst.pt"
FINETUNED_CHECKPOINT = "checkpoints/finetuned_patchtst.pt"
NORM_STATS_DATASET = "etth1"
OUTPUT_DIR = "evaluation/results"
JSON_OUTPUT_PATH = "evaluation/results/final_metrics.json"


def load_etth1_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load and split ETTh1 data into train/val/test with normalization stats.

    Returns:
        Tuple of (train_data, val_data, test_data, norm_stats) where:
            - train_data: 1D numpy array of raw training values
            - val_data: 1D numpy array of raw validation values
            - test_data: 1D numpy array of raw test values
            - norm_stats: Dictionary with "mean" and "std" keys
    """
    print(f"[Pipeline] Loading ETTh1 data from: {ETTH1_DATA_PATH}")

    df = pd.read_csv(ETTH1_DATA_PATH)

    # Use the first numeric column as the target (OT — Oil Temperature)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = numeric_cols[0] if numeric_cols else df.columns[1]
    raw_data = df[target_col].values.astype(np.float64)

    # Split chronologically into train/val/test (70/15/15)
    train_data, val_data, test_data = split_chronological(raw_data)

    # Load normalization statistics
    norm_stats = load_normalization_stats(NORM_STATS_DATASET)

    print(f"[Pipeline] Data loaded — Train: {len(train_data)}, "
          f"Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data, norm_stats


def run_baselines(
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> dict[str, dict]:
    """Run all three classical baselines with timing instrumentation.

    Args:
        train_data: 1D numpy array of raw (original scale) training data.
        test_data: 1D numpy array of raw (original scale) test data.

    Returns:
        Dictionary mapping model names to their result dictionaries containing
        metrics and inference_time.
    """
    results = {}

    # --- Naive Baseline ---
    print("\n" + "=" * 60)
    print("RUNNING NAIVE BASELINE")
    print("=" * 60)

    naive_result = run_naive_baseline(
        test_data=test_data,
        context_length=CONTEXT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
    )
    results["Naive"] = {
        "mae": naive_result["metrics"]["mae"],
        "mse": naive_result["metrics"]["mse"],
        "mase": naive_result["metrics"]["mase"],
        "crps": None,  # Point forecast only
        "inference_time": naive_result["inference_time"],
    }
    print(f"[Naive] MAE: {naive_result['metrics']['mae']:.4f}, "
          f"MSE: {naive_result['metrics']['mse']:.4f}, "
          f"MASE: {naive_result['metrics']['mase']:.4f}, "
          f"Time: {naive_result['inference_time']:.2f}s")

    # --- ARIMA Baseline ---
    print("\n" + "=" * 60)
    print("RUNNING ARIMA BASELINE")
    print("=" * 60)

    arima_result = run_arima_baseline_eval(
        train=train_data,
        test_data=test_data,
        context_length=CONTEXT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
    )
    results["ARIMA"] = {
        "mae": arima_result["metrics"]["mae"],
        "mse": arima_result["metrics"]["mse"],
        "mase": arima_result["metrics"]["mase"],
        "crps": None,  # Point forecast only
        "inference_time": arima_result["inference_time"],
    }

    # --- Prophet Baseline ---
    print("\n" + "=" * 60)
    print("RUNNING PROPHET BASELINE")
    print("=" * 60)

    prophet_result = run_prophet_baseline_eval(
        train=train_data,
        test_data=test_data,
        context_length=CONTEXT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
    )
    results["Prophet"] = {
        "mae": prophet_result["metrics"]["mae"],
        "mse": prophet_result["metrics"]["mse"],
        "mase": prophet_result["metrics"]["mase"],
        "crps": None,  # Point forecast only
        "inference_time": prophet_result["inference_time"],
    }

    return results


def run_zero_shot(
    test_data: np.ndarray,
    train_data: np.ndarray,
    norm_stats: dict,
    device: str = "cpu",
) -> dict[str, float]:
    """Run zero-shot evaluation with timing instrumentation.

    Args:
        test_data: 1D numpy array of normalized test data.
        train_data: 1D numpy array of raw training data (for MASE).
        norm_stats: Normalization statistics dictionary.
        device: Device for inference ("cpu" or "cuda").

    Returns:
        Metric dictionary with mae, mse, mase, crps, inference_time.
    """
    print("\n" + "=" * 60)
    print("RUNNING ZERO-SHOT EVALUATION")
    print("=" * 60)

    zs_result = run_zero_shot_evaluation(
        checkpoint_path=PRETRAINED_CHECKPOINT,
        test_data=test_data,
        norm_stats=norm_stats,
        train_data=train_data,
        context_length=CONTEXT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        device=device,
        output_dir="forecasting/results",
    )

    metrics = zs_result["metrics"]
    print(f"[Zero-Shot] MAE: {metrics['mae']:.4f}, "
          f"MSE: {metrics['mse']:.4f}, "
          f"MASE: {metrics['mase']:.4f}, "
          f"CRPS: {metrics['crps']:.4f}, "
          f"Time: {metrics['inference_time']:.2f}s")

    return metrics


def run_finetune(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    norm_stats: dict,
    device: str = "cpu",
) -> tuple[dict[str, float], list[float]]:
    """Run fine-tune evaluation with timing instrumentation.

    Args:
        train_data: 1D numpy array of normalized training data.
        val_data: 1D numpy array of normalized validation data.
        test_data: 1D numpy array of normalized test data.
        norm_stats: Normalization statistics dictionary.
        device: Device for training/inference ("cpu" or "cuda").

    Returns:
        Tuple of (metrics_dict, train_losses) where metrics_dict contains
        mae, mse, mase, crps, inference_time and train_losses is per-epoch.
    """
    print("\n" + "=" * 60)
    print("RUNNING FINE-TUNE EVALUATION (10 epochs, AdamW lr=5e-5)")
    print("=" * 60)

    ft_result = run_finetune_evaluation(
        pretrained_checkpoint_path=PRETRAINED_CHECKPOINT,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        norm_stats=norm_stats,
        device=device,
        save_dir="checkpoints",
    )

    metrics = ft_result["metrics"]
    print(f"[Fine-Tune] Completed {ft_result['epochs_completed']} epochs")
    print(f"[Fine-Tune] MAE: {metrics['mae']:.4f}, "
          f"MSE: {metrics['mse']:.4f}, "
          f"MASE: {metrics['mase']:.4f}, "
          f"CRPS: {metrics['crps']:.4f}, "
          f"Time: {metrics['inference_time']:.2f}s")

    return metrics, ft_result.get("train_losses", [])


def generate_plots(
    all_results: dict[str, dict],
    zs_result: dict | None = None,
    train_losses: list[float] | None = None,
) -> None:
    """Generate all 3 publication-quality plots.

    Args:
        all_results: Complete results dictionary with all model metrics.
        zs_result: Zero-shot evaluation result (with forecasts/actuals for plot).
        train_losses: Per-epoch training losses for loss curve plot.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # --- Plot 1: Forecast plot (zero-shot predictions vs actuals) ---
    if zs_result is not None and "forecasts" in zs_result and "actuals" in zs_result:
        forecasts = zs_result["forecasts"]
        actuals = zs_result["actuals"]

        # Plot the first window as a representative example
        if len(forecasts) > 0:
            p10 = forecasts[0, :, 0]
            p50 = forecasts[0, :, 1]
            p90 = forecasts[0, :, 2]
            actual = actuals[0]

            path = plot_forecast(
                actual=actual,
                p50=p50,
                p10=p10,
                p90=p90,
                window_index=0,
                dataset_name="ETTh1",
                output_dir=OUTPUT_DIR,
                dpi=300,
            )
            print(f"[Plots] Forecast plot saved: {path}")

    # --- Plot 2: Pretraining loss curve ---
    # Use available training loss data; if fine-tune losses are available,
    # show them as a single domain. For pretraining losses, check if a
    # pretraining log exists.
    domain_losses = _load_pretraining_losses()
    if domain_losses:
        path = plot_loss_curve(
            domain_losses=domain_losses,
            output_dir=OUTPUT_DIR,
            dpi=300,
        )
        print(f"[Plots] Loss curve saved: {path}")
    else:
        # If no pretraining logs found, generate with empty data
        path = plot_loss_curve(
            domain_losses={"Energy": [], "Weather": [], "Finance": []},
            output_dir=OUTPUT_DIR,
            dpi=300,
        )
        print(f"[Plots] Loss curve saved (no pretraining data available): {path}")

    # --- Plot 3: MAE comparison bar chart ---
    model_maes = {}
    for model_name, metrics in all_results.items():
        # Map model names to the format expected by plot_mae_bar_chart
        if model_name == "PatchTST (zero-shot)":
            model_maes["PatchTST zero-shot"] = metrics["mae"]
        elif model_name == "PatchTST (fine-tuned)":
            model_maes["PatchTST fine-tuned"] = metrics["mae"]
        else:
            model_maes[model_name] = metrics["mae"]

    path = plot_mae_bar_chart(
        model_maes=model_maes,
        output_dir=OUTPUT_DIR,
        dpi=300,
    )
    print(f"[Plots] MAE bar chart saved: {path}")


def _load_pretraining_losses() -> dict[str, list[float]]:
    """Attempt to load pretraining loss history from checkpoint or log files.

    Returns:
        Dictionary mapping domain names to per-epoch loss lists.
        Returns empty dict if no pretraining data is available.
    """
    import torch

    # Try loading from the pretrained checkpoint which may contain loss history
    if os.path.isfile(PRETRAINED_CHECKPOINT):
        try:
            checkpoint = torch.load(
                PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False
            )
            # Check for domain-specific loss history in checkpoint
            if "domain_losses" in checkpoint:
                return checkpoint["domain_losses"]
            if "train_losses" in checkpoint:
                # If only aggregate losses, assign to a generic domain
                losses = checkpoint["train_losses"]
                if isinstance(losses, dict):
                    return losses
                # Single list — cannot split by domain
                return {"Energy": losses, "Weather": [], "Finance": []}
        except Exception:
            pass

    return {}


def main() -> None:
    """Execute the full unified evaluation pipeline.

    Orchestrates all evaluation steps in sequence:
        1. Load ETTh1 data
        2. Run Naive, ARIMA, Prophet baselines
        3. Run zero-shot evaluation
        4. Run fine-tune evaluation
        5. Print results table and save JSON
        6. Generate all 3 plots
    """
    import torch

    pipeline_start = time.time()

    print("=" * 60)
    print("UNIFIED EVALUATION PIPELINE")
    print(f"Context Length: {CONTEXT_LENGTH}, Forecast Horizon: {FORECAST_HORIZON}, "
          f"Stride: {STRIDE}")
    print("=" * 60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Using device: {device}")

    # -----------------------------------------------------------------------
    # Step 1: Load ETTh1 data
    # -----------------------------------------------------------------------
    train_data, val_data, test_data, norm_stats = load_etth1_data()

    # Convert to original scale for baselines (they operate on raw values)
    train_original = inverse_normalize(train_data, norm_stats)
    test_original = inverse_normalize(test_data, norm_stats)

    # Collect all results
    all_results: dict[str, dict] = {}
    zs_full_result = None
    ft_train_losses: list[float] = []

    # -----------------------------------------------------------------------
    # Step 2: Run classical baselines (Naive, ARIMA, Prophet)
    # -----------------------------------------------------------------------
    baseline_results = run_baselines(
        train_data=train_original,
        test_data=test_original,
    )
    all_results.update(baseline_results)

    # -----------------------------------------------------------------------
    # Step 3: Run zero-shot evaluation
    # -----------------------------------------------------------------------
    if os.path.isfile(PRETRAINED_CHECKPOINT):
        zs_metrics = run_zero_shot(
            test_data=test_data,
            train_data=train_original,
            norm_stats=norm_stats,
            device=device,
        )
        all_results["PatchTST (zero-shot)"] = zs_metrics

        # Also get the full result with forecasts/actuals for plotting
        # (run_zero_shot_evaluation already ran, re-use its output)
        # Re-run to capture forecasts and actuals for visualization
        try:
            zs_full_result = run_zero_shot_evaluation(
                checkpoint_path=PRETRAINED_CHECKPOINT,
                test_data=test_data,
                norm_stats=norm_stats,
                train_data=train_original,
                context_length=CONTEXT_LENGTH,
                forecast_horizon=FORECAST_HORIZON,
                stride=STRIDE,
                device=device,
                output_dir="forecasting/results",
            )
        except Exception as e:
            print(f"[Pipeline] Warning: Could not retrieve forecast arrays for plotting: {e}")
    else:
        print(f"\n[Pipeline] Pretrained checkpoint not found at '{PRETRAINED_CHECKPOINT}'. "
              f"Skipping zero-shot evaluation.")

    # -----------------------------------------------------------------------
    # Step 4: Run fine-tune evaluation
    # -----------------------------------------------------------------------
    if os.path.isfile(PRETRAINED_CHECKPOINT):
        try:
            ft_metrics, ft_train_losses = run_finetune(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                norm_stats=norm_stats,
                device=device,
            )
            all_results["PatchTST (fine-tuned)"] = ft_metrics
        except Exception as e:
            print(f"\n[Pipeline] Fine-tune evaluation failed: {e}")
            print("[Pipeline] Continuing without fine-tune results.")
    else:
        print(f"\n[Pipeline] Pretrained checkpoint not found at '{PRETRAINED_CHECKPOINT}'. "
              f"Skipping fine-tune evaluation.")

    # -----------------------------------------------------------------------
    # Step 5: Print results table and save JSON
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60 + "\n")

    print_results_table(all_results)
    save_results_json(all_results, output_path=JSON_OUTPUT_PATH)
    print(f"\n[Pipeline] Results saved to: {JSON_OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Step 6: Generate all 3 plots
    # -----------------------------------------------------------------------
    generate_plots(
        all_results=all_results,
        zs_result=zs_full_result,
        train_losses=ft_train_losses,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    pipeline_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — Total time: {pipeline_time:.1f}s")
    print(f"Results JSON: {JSON_OUTPUT_PATH}")
    print(f"Plots directory: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
