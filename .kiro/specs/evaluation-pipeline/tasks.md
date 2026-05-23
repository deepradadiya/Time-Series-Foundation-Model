# Implementation Plan: Evaluation Pipeline

## Overview

This plan implements the complete evaluation pipeline for the Time-Series Foundation Model. It builds on existing partial implementations in `evaluation/baselines.py`, `evaluation/evaluate.py`, `evaluation/metrics.py`, and `evaluation/visualize.py`, extending them with a Naive baseline, dedicated zero-shot and fine-tune evaluation scripts, a results table module, and publication-quality visualizations. The implementation uses Python with numpy, torch, pmdarima, prophet, and matplotlib.

## Tasks

- [x] 1. Implement Naive baseline and extend forecasting/baselines.py
  - [x] 1.1 Create `forecasting/baselines.py` with Naive, ARIMA, and Prophet baseline runners
    - Implement `naive_forecast(context, horizon=96)` that returns an array of `context[-1]` repeated for `horizon` steps
    - Implement `run_naive_baseline(test_data, context_length=512, forecast_horizon=96, stride=96)` with timing instrumentation returning `{"forecasts", "metrics", "inference_time"}`
    - Implement `run_arima_baseline_eval(train, test_data, context_length=512, forecast_horizon=96, stride=96)` wrapping existing `evaluation/baselines.py` ARIMA logic with pmdarima `auto_arima(max_p=5, max_d=2, max_q=5)`, fallback to seasonal naive (period=24) on failure
    - Implement `run_prophet_baseline_eval(train, test_data, context_length=512, forecast_horizon=96, stride=96)` wrapping existing Prophet logic with `daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False`, fallback to seasonal naive (period=24) on failure
    - Raise `ValueError` if context is empty (length 0)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 1.2 Write property test for Naive forecast constant output
    - **Property 1: Naive forecast produces constant array of last context value**
    - **Validates: Requirements 1.1**

  - [ ]* 1.3 Write property test for baseline forecast output length invariant
    - **Property 2: Baseline forecast output length invariant**
    - **Validates: Requirements 2.2, 3.2**

  - [ ]* 1.4 Write unit tests for forecasting/baselines.py
    - Test empty context raises ValueError
    - Test ARIMA fallback to seasonal naive on fitting failure
    - Test Prophet fallback to seasonal naive on fitting failure
    - Test naive forecast on known input produces expected constant array
    - _Requirements: 1.2, 2.5, 3.5_

- [x] 2. Extend evaluation/metrics.py with documentation and validation
  - [x] 2.1 Validate and document existing metric functions in `evaluation/metrics.py`
    - Verify `mae`, `mse`, `mase`, `crps_quantile` implementations match design specifications
    - Add inline comments stating mathematical formula and plain-language explanation for each function
    - Ensure `mase` uses `seasonal_period=24` as default
    - Ensure `mase` returns `float("inf")` when naive baseline MAE is zero
    - Ensure `crps_quantile` computes `(2/K) * sum(pinball_loss)` with quantiles [0.1, 0.5, 0.9]
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 2.2 Write property test for MAE reference computation
    - **Property 4: MAE equals reference computation**
    - **Validates: Requirements 6.1**

  - [ ]* 2.3 Write property test for MSE non-negative and reference computation
    - **Property 5: MSE is non-negative and equals reference computation**
    - **Validates: Requirements 6.2**

  - [ ]* 2.4 Write property test for MASE scaling relationship
    - **Property 6: MASE scaling relationship**
    - **Validates: Requirements 6.3**

  - [ ]* 2.5 Write property test for CRPS pinball formula
    - **Property 7: CRPS equals scaled pinball loss sum**
    - **Validates: Requirements 6.5**

  - [ ]* 2.6 Write property test for metric determinism
    - **Property 8: Metric functions are deterministic**
    - **Validates: Requirements 6.7**

  - [ ]* 2.7 Write unit tests for edge cases in evaluation/metrics.py
    - Test MASE returns inf for zero denominator (constant seasonal series)
    - Test MAE of identical arrays is 0.0
    - Test MSE of identical arrays is 0.0
    - _Requirements: 6.4_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement zero-shot transfer evaluation
  - [x] 4.1 Create `forecasting/zero_shot_eval.py` with `run_zero_shot_evaluation`
    - Load pretrained checkpoint and freeze all encoder params (`requires_grad=False`)
    - Attach `ProbabilisticForecastHead` with random weights (d_model=256, num_patches=63, forecast_horizon=96, quantiles=[0.1, 0.5, 0.9])
    - Run sliding window inference (context=512, horizon=96, stride=96)
    - Compute MAE, MSE, MASE (P50 as point forecast), CRPS (P10/P50/P90)
    - Measure inference time (wall-clock seconds)
    - Save predictions to `forecasting/results/zero_shot_predictions.csv` with columns: window_index, time_step, actual, P10, P50, P90
    - Raise `FileNotFoundError` if checkpoint does not exist
    - Create output directory if it does not exist
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_

  - [ ]* 4.2 Write property test for zero-shot output shape
    - **Property 3: Zero-shot inference output shape**
    - **Validates: Requirements 4.5**

  - [ ]* 4.3 Write unit tests for zero-shot evaluation
    - Test missing checkpoint raises FileNotFoundError
    - Test that encoder weights are frozen (no gradient computation)
    - _Requirements: 4.2_

- [x] 5. Implement fine-tune evaluation
  - [x] 5.1 Create `forecasting/finetune_eval.py` with `run_finetune_evaluation`
    - Load pretrained checkpoint and unfreeze ALL encoder layers
    - Attach `ProbabilisticForecastHead`
    - Train with AdamW(lr=5e-5), batch_size=32, 10 epochs on ETTh1 train split
    - Halt on NaN loss (raise `RuntimeError` reporting epoch number)
    - Save fine-tuned checkpoint to `checkpoints/` directory
    - Evaluate on test split: MAE, MSE, MASE, CRPS
    - Measure inference time on test set
    - Return `{"metrics", "train_losses", "val_losses", "epochs_completed", "checkpoint_path"}`
    - Raise `FileNotFoundError` if pretrained checkpoint does not exist
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 5.2 Write unit tests for fine-tune evaluation
    - Test missing checkpoint raises FileNotFoundError
    - Test NaN loss halts training and reports epoch number
    - _Requirements: 5.6, 5.7_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement results comparison table and JSON export
  - [x] 7.1 Create `evaluation/results_table.py` with table formatting and JSON export
    - Implement `format_results_table(results)` producing a formatted string with columns: Model, MAE, MSE, MASE, CRPS, Inference_Time
    - Row order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned)
    - Display CRPS as "N/A" for point-forecast baselines (Naive, ARIMA, Prophet)
    - Display Inference_Time with human-readable units (<1ms, ~Xs, ~Xms)
    - Implement `print_results_table(results)` printing to stdout
    - Implement `save_results_json(results, output_path="evaluation/results/final_metrics.json")` saving JSON with model names as keys, metrics rounded to 4 decimal places
    - Create output directory with `os.makedirs(exist_ok=True)` if it does not exist
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write property test for JSON round-trip
    - **Property 9: JSON results round-trip preserves structure and precision**
    - **Validates: Requirements 7.3**

  - [ ]* 7.3 Write unit tests for results_table.py
    - Test row order matches specification (Naive, ARIMA, Prophet, PatchTST zero-shot, PatchTST fine-tuned)
    - Test CRPS shows "N/A" for baselines
    - Test JSON output has values rounded to 4 decimal places
    - _Requirements: 7.2, 7.4_

- [x] 8. Implement publication-quality visualizations
  - [x] 8.1 Create `evaluation/visualize_forecasts.py` with three plot functions
    - Implement `plot_forecast(actual, p50, p10, p90, window_index, dataset_name="ETTh1", output_dir="evaluation/results", dpi=300)`:
      - Actual as solid line, P50 as dashed line, P10-P90 shaded (alpha 0.2-0.4)
      - Include x-axis (time steps), y-axis (value), legend, title with window index and dataset name
      - Save as PNG at minimum 300 DPI
      - Raise ValueError if array lengths differ
    - Implement `plot_loss_curve(domain_losses, output_dir="evaluation/results", dpi=300)`:
      - Separate lines for Energy, Weather, Finance domains with distinct colors and legend
      - Omit domains with zero epochs of data
      - Save to `evaluation/results/pretraining_loss_curve.png`
    - Implement `plot_mae_bar_chart(model_maes, output_dir="evaluation/results", dpi=300)`:
      - Bars in order: Naive, ARIMA, Prophet, PatchTST zero-shot, PatchTST fine-tuned
      - One color for baselines, different color for PatchTST models
      - Value labels on top (4 decimal places)
      - Save to `evaluation/results/mae_comparison_bar_chart.png`
    - All plots use non-interactive matplotlib backend (Agg) for file-only output
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 8.2 Write property test for loss curve domain omission
    - **Property 10: Loss curve gracefully omits empty domains**
    - **Validates: Requirements 9.4**

  - [ ]* 8.3 Write unit tests for visualize_forecasts.py
    - Test forecast plot dimension mismatch raises ValueError
    - Test plot saved at minimum 300 DPI
    - Test loss curve with all-empty domains produces no error
    - _Requirements: 8.5, 8.3_

- [x] 9. Wire all components together
  - [x] 9.1 Integrate all modules into a unified evaluation pipeline entry point
    - Update or create a top-level orchestration script that:
      1. Runs Naive, ARIMA, Prophet baselines via `forecasting/baselines.py`
      2. Runs zero-shot evaluation via `forecasting/zero_shot_eval.py`
      3. Runs fine-tune evaluation via `forecasting/finetune_eval.py`
      4. Computes all metrics via `evaluation/metrics.py`
      5. Prints results table and saves JSON via `evaluation/results_table.py`
      6. Generates all 3 plots via `evaluation/visualize_forecasts.py`
    - Ensure consistent windowing (context=512, horizon=96, stride=96) across all models
    - Wire timing instrumentation for all models
    - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.4, 4.6, 4.7, 4.8, 5.4, 5.5, 7.1, 7.3_

  - [ ]* 9.2 Write integration tests for the full pipeline
    - Test full naive evaluation on sample data produces expected output structure
    - Test results table + JSON save produces valid file with all 5 model keys
    - Test all 3 plots are generated as PNG files in the output directory
    - _Requirements: 1.3, 7.1, 7.3, 8.3, 9.3, 10.5_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python with Hypothesis for property-based testing (`.hypothesis/` directory already exists)
- Existing code in `evaluation/baselines.py`, `evaluation/evaluate.py`, `evaluation/metrics.py`, and `evaluation/visualize.py` should be reused and extended, not replaced
- Key parameters: context_length=512, forecast_horizon=96, stride=96, d_model=256, num_patches=63, quantiles=[0.1, 0.5, 0.9], finetune lr=5e-5, batch_size=32, 10 epochs, MASE seasonal_period=24, plots at 300 DPI minimum

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"] },
    { "id": 2, "tasks": ["4.1", "5.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.2"] },
    { "id": 4, "tasks": ["7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "8.2", "8.3"] },
    { "id": 6, "tasks": ["9.1"] },
    { "id": 7, "tasks": ["9.2"] }
  ]
}
```
