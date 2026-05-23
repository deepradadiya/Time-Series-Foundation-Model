"""Results comparison table and JSON export for the evaluation pipeline.

This module formats evaluation results into a human-readable comparison table
and persists them as JSON for downstream consumption. It handles all 5 models:
Naive, ARIMA, Prophet, PatchTST (zero-shot), and PatchTST (fine-tuned).

Related modules:
    - evaluation/metrics.py: Provides metric computation (MAE, MSE, MASE, CRPS)
    - forecasting/baselines.py: Produces baseline metric dictionaries
    - forecasting/zero_shot_eval.py: Produces zero-shot metric dictionaries
    - forecasting/finetune_eval.py: Produces fine-tune metric dictionaries
"""

import json
import os


# Canonical row order for the results table
MODEL_ORDER = [
    "Naive",
    "ARIMA",
    "Prophet",
    "PatchTST (zero-shot)",
    "PatchTST (fine-tuned)",
]

# Models that only produce point forecasts (no CRPS)
POINT_FORECAST_MODELS = {"Naive", "ARIMA", "Prophet"}


def _format_inference_time(seconds: float) -> str:
    """Format inference time with human-readable units.

    Rules:
        - < 0.001s  → "<1ms"
        - >= 0.001s and < 1s → "~Xms" (e.g., "~234ms")
        - >= 1s → "~Xs" (e.g., "~45s")

    Args:
        seconds: Wall-clock inference time in seconds.

    Returns:
        Human-readable string representation.
    """
    if seconds < 0.001:
        return "<1ms"
    elif seconds < 1.0:
        ms = int(round(seconds * 1000))
        return f"~{ms}ms"
    else:
        s = int(round(seconds))
        return f"~{s}s"


def format_results_table(results: dict[str, dict[str, float]]) -> str:
    """Format a comparison table string with columns: Model, MAE, MSE, MASE, CRPS, Inference_Time.

    Row order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned).
    CRPS shows "N/A" for point-forecast baselines (Naive, ARIMA, Prophet).
    Inference_Time uses human-readable units (<1ms, ~Xs, ~Xms).

    Args:
        results: Dictionary mapping model names to metric dictionaries.
            Each metric dictionary contains keys: mae, mse, mase, crps, inference_time.
            crps may be None for point-forecast baselines.

    Returns:
        Formatted table as a multi-line string.
    """
    # Column headers
    headers = ["Model", "MAE", "MSE", "MASE", "CRPS", "Inference_Time"]

    # Build rows in canonical order
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        metrics = results[model_name]

        mae_str = f"{metrics['mae']:.4f}"
        mse_str = f"{metrics['mse']:.4f}"
        mase_str = f"{metrics['mase']:.4f}"

        # CRPS is N/A for point-forecast-only baselines
        if model_name in POINT_FORECAST_MODELS or metrics.get("crps") is None:
            crps_str = "N/A"
        else:
            crps_str = f"{metrics['crps']:.4f}"

        time_str = _format_inference_time(metrics["inference_time"])

        rows.append([model_name, mae_str, mse_str, mase_str, crps_str, time_str])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header line
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "  ".join("-" * col_widths[i] for i in range(len(headers)))

    # Format data rows
    data_lines = []
    for row in rows:
        line = "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        data_lines.append(line)

    # Assemble table
    table_parts = [header_line, separator] + data_lines
    return "\n".join(table_parts)


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print the formatted comparison table to stdout.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
    """
    print(format_results_table(results))


def save_results_json(
    results: dict[str, dict[str, float]],
    output_path: str = "evaluation/results/final_metrics.json",
) -> None:
    """Save results as JSON with model names as keys, metrics rounded to 4 decimal places.

    Creates the output directory if it does not exist.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
            Each metric dictionary contains keys: mae, mse, mase, crps, inference_time.
        output_path: Path to the output JSON file.
            Defaults to "evaluation/results/final_metrics.json".
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Round numeric values to 4 decimal places, preserve None for crps
    rounded_results = {}
    for model_name, metrics in results.items():
        rounded_metrics = {}
        for key, value in metrics.items():
            if value is None:
                rounded_metrics[key] = None
            else:
                rounded_metrics[key] = round(value, 4)
        rounded_results[model_name] = rounded_metrics

    with open(output_path, "w") as f:
        json.dump(rounded_results, f, indent=4)
