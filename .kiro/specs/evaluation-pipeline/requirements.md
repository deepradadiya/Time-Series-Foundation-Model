# Requirements Document

## Introduction

This document specifies the complete evaluation pipeline for the Time-Series Foundation Model project. The pipeline demonstrates the model's zero-shot transfer learning capabilities by comparing a pretrained PatchTST backbone against classical baselines (Naive, ARIMA, Prophet) on the ETTh1 benchmark dataset. It also includes a fine-tuning evaluation to establish the upper bound of performance. The pipeline produces metrics (MAE, MSE, MASE, CRPS), a comparison results table, prediction CSVs, and publication-quality visualizations.

## Glossary

- **Evaluation_Pipeline**: The complete system that orchestrates baseline forecasting, zero-shot evaluation, fine-tune evaluation, metric computation, results aggregation, and visualization generation.
- **Baseline_Forecaster**: A module in `forecasting/baselines.py` that implements classical forecasting methods (Naive, ARIMA, Prophet) for comparison against the foundation model.
- **Naive_Baseline**: A forecasting method that predicts the last known value of the input context for all future time steps, serving as the minimum performance floor.
- **ARIMA_Baseline**: A statistical forecasting method using the pmdarima library's `auto_arima()` function to automatically select optimal (p,d,q) parameters and generate point forecasts.
- **Prophet_Baseline**: A forecasting method using Facebook Prophet library that decomposes time series into trend and seasonality components to generate point forecasts.
- **Zero_Shot_Evaluator**: A module in `forecasting/zero_shot_eval.py` that loads the pretrained PatchTST backbone with a randomly initialized LinearForecastHead and generates probabilistic forecasts on ETTh1 without any training on ETTh1 data.
- **LinearForecastHead**: A single linear layer that maps from d_model (256) to forecast_horizon multiplied by 3 (for P10/P50/P90 probabilistic output), attached to the frozen pretrained backbone.
- **Finetune_Evaluator**: A module in `forecasting/finetune_eval.py` that unfreezes all pretrained backbone layers and fine-tunes the entire model on the ETTh1 train split for 10 epochs with lr=5e-5 to establish the upper bound of performance.
- **Metrics_Module**: A module in `evaluation/metrics.py` that implements MAE, MSE, MASE, and CRPS metric computations with clear explanations.
- **Results_Table**: A module in `evaluation/results_table.py` that prints a formatted comparison table and saves results to `evaluation/results/final_metrics.json`.
- **Visualization_Module**: A module in `evaluation/visualize_forecasts.py` that generates three publication-quality plots saved as high-resolution PNG files to `evaluation/results/`.
- **ETTh1_Test_Set**: The test split (last 15%) of the ETTh1 hourly electricity transformer temperature dataset used for all evaluations.
- **Forecast_Horizon**: The number of future time steps predicted, fixed at 96 steps for all models.
- **MASE**: Mean Absolute Scaled Error — MAE divided by the MAE of the Naive baseline; a value less than 1 means the model beats the Naive baseline.
- **CRPS**: Continuous Ranked Probability Score — measures the quality of probabilistic predictions using P10/P50/P90 quantiles; lower values indicate better calibrated prediction intervals.
- **Inference_Time**: Wall-clock time in seconds measured for each model to generate forecasts on the entire ETTh1 test set.

## Requirements

### Requirement 1: Naive Baseline Forecaster

**User Story:** As a researcher, I want a Naive baseline that predicts the last known value for all future steps, so that I have a sanity-check performance floor for comparison.

#### Acceptance Criteria

1. WHEN the Naive_Baseline receives a context window of at least 1 historical value, THE Baseline_Forecaster SHALL produce a forecast array of length Forecast_Horizon (96 time steps) where every element equals the final value of the context window.
2. IF the Naive_Baseline receives an empty context window (length 0), THEN THE Baseline_Forecaster SHALL raise an error indicating that the context window must contain at least one value.
3. WHEN the Naive_Baseline is evaluated on the ETTh1_Test_Set using a sliding window approach with Context_Length of 512 and stride equal to Forecast_Horizon (96), THE Baseline_Forecaster SHALL compute and return a dictionary containing MAE, MSE, and MASE metrics averaged across all test windows.
4. WHEN the Naive_Baseline is evaluated, THE Baseline_Forecaster SHALL measure and record the wall-clock Inference_Time in seconds (to at least 2 decimal places) for generating all test set forecasts.
5. THE Baseline_Forecaster SHALL use the Naive_Baseline MAE (mean absolute error of the naive constant-last-value forecast across all test windows) as the denominator when computing MASE for all other models evaluated in the pipeline, so that a MASE value below 1 indicates improvement over the Naive floor.

### Requirement 2: ARIMA Baseline Forecaster

**User Story:** As a researcher, I want an ARIMA baseline using pmdarima's auto_arima to automatically find optimal parameters, so that I can compare the foundation model against a well-tuned statistical method.

#### Acceptance Criteria

1. WHEN the ARIMA_Baseline receives a context window of 512 time steps, THE Baseline_Forecaster SHALL use pmdarima's `auto_arima()` function to automatically determine the best (p,d,q) parameters for that window with search bounds max_p=5, max_d=2, max_q=5.
2. WHEN the ARIMA_Baseline generates a forecast, THE Baseline_Forecaster SHALL produce a point forecast array of length Forecast_Horizon (96 steps) for each sliding window in the ETTh1_Test_Set, using the same test windows as the foundation model evaluation.
3. WHEN the ARIMA_Baseline completes evaluation across all test windows, THE Baseline_Forecaster SHALL compute and record MAE, MSE, and MASE metrics averaged across all test windows and all forecast steps, using a seasonal period of 24 for the MASE scaling denominator.
4. WHEN the ARIMA_Baseline completes evaluation, THE Baseline_Forecaster SHALL measure and record the total Inference_Time in seconds, covering the cumulative duration of fitting and forecasting across all test windows.
5. IF `auto_arima()` raises any exception during fitting or forecasting for a given window, THEN THE Baseline_Forecaster SHALL fall back to the seasonal naive method with period 24 for that window and log a warning message to stdout indicating the window index and the exception type.

### Requirement 3: Prophet Baseline Forecaster

**User Story:** As a researcher, I want a Prophet baseline using Facebook's Prophet library, so that I can compare the foundation model against a modern decomposition-based forecasting tool.

#### Acceptance Criteria

1. WHEN the Prophet_Baseline receives a context window of 512 time steps, THE Baseline_Forecaster SHALL fit a Prophet model with daily seasonality enabled, weekly seasonality enabled, and yearly seasonality disabled on the historical data consisting of training data concatenated with the context portion.
2. WHEN the Prophet_Baseline generates a forecast, THE Baseline_Forecaster SHALL produce a point forecast array of length 96 steps for each sliding window in the ETTh1_Test_Set, using the same window positions (context_length=512, stride=96) as the foundation model evaluation.
3. WHEN the Prophet_Baseline completes evaluation, THE Baseline_Forecaster SHALL compute MAE, MSE, and MASE metrics averaged across all test windows, where MASE uses a seasonal naive scaling denominator with period 24.
4. WHEN the Prophet_Baseline completes evaluation, THE Baseline_Forecaster SHALL measure and record the total wall-clock Inference_Time in seconds, covering all test windows from first fit to last forecast completion.
5. IF Prophet fails to fit or predict for a given window, THEN THE Baseline_Forecaster SHALL fall back to the seasonal naive method with period 24 for that window and log a warning message indicating the window index and the error type that caused the failure.
6. WHEN the Prophet_Baseline completes evaluation, THE Baseline_Forecaster SHALL output the computed metrics (MAE, MSE, MASE) and Inference_Time to the console and include them in the results dictionary returned to the evaluation pipeline.

### Requirement 4: Zero-Shot Transfer Evaluation

**User Story:** As a researcher, I want to evaluate the pretrained PatchTST backbone on ETTh1 without any ETTh1-specific training, so that I can demonstrate genuine cross-domain transfer learning capabilities.

#### Acceptance Criteria

1. WHEN the Zero_Shot_Evaluator is invoked, THE Zero_Shot_Evaluator SHALL load the pretrained PatchTST backbone from the checkpoint file and freeze all encoder weights by disabling gradient computation so that no parameter updates occur during inference.
2. IF the pretrained checkpoint file does not exist or cannot be loaded, THEN THE Zero_Shot_Evaluator SHALL raise an error indicating the missing or invalid checkpoint path without producing partial results.
3. WHEN the Zero_Shot_Evaluator is invoked, THE Zero_Shot_Evaluator SHALL attach a ProbabilisticForecastHead with randomly initialized weights that maps encoder output (d_model=256, num_patches=63) to forecast_horizon (96) multiplied by 3 outputs for P10, P50, and P90 quantile prediction.
4. THE Zero_Shot_Evaluator SHALL NOT perform any gradient-based parameter updates on the ProbabilisticForecastHead using ETTh1 data before generating forecasts.
5. WHEN the Zero_Shot_Evaluator runs inference, THE Zero_Shot_Evaluator SHALL use a sliding window with context_length=512, forecast_horizon=96, and stride=96 to generate P10, P50, and P90 quantile forecasts for each non-overlapping test window across the entire ETTh1_Test_Set.
6. WHEN the Zero_Shot_Evaluator completes inference, THE Zero_Shot_Evaluator SHALL compute MAE, MSE, and MASE (with seasonal_period=24) using the P50 quantile as the point forecast.
7. WHEN the Zero_Shot_Evaluator completes inference, THE Zero_Shot_Evaluator SHALL compute CRPS using all three quantiles (P10, P50, P90) with quantile levels [0.1, 0.5, 0.9] to assess probabilistic forecast quality.
8. WHEN the Zero_Shot_Evaluator completes inference, THE Zero_Shot_Evaluator SHALL measure and record the Inference_Time in seconds, covering the duration from the start of the first forward pass to the completion of the last forward pass across all test windows.
9. WHEN the Zero_Shot_Evaluator completes inference, THE Zero_Shot_Evaluator SHALL create the output directory if it does not exist and save all predictions and actual values to `forecasting/results/zero_shot_predictions.csv` with columns: window_index, time_step, actual, P10, P50, P90.

### Requirement 5: Fine-Tune Evaluation

**User Story:** As a researcher, I want to fine-tune the pretrained backbone on ETTh1 training data for 10 epochs, so that I can establish the upper bound of performance and demonstrate the foundation model's value proposition (zero-shot is good, fine-tuned is better).

#### Acceptance Criteria

1. WHEN the Finetune_Evaluator is invoked, THE Finetune_Evaluator SHALL load the pretrained PatchTST backbone and unfreeze all encoder layers for gradient updates.
2. WHEN the Finetune_Evaluator trains the model, THE Finetune_Evaluator SHALL use the AdamW optimizer with a learning rate of 5e-5 and a batch size of 32 to preserve pretrained representations while adapting to ETTh1.
3. WHEN the Finetune_Evaluator trains the model, THE Finetune_Evaluator SHALL train for exactly 10 epochs on the ETTh1 train split.
4. WHEN the Finetune_Evaluator completes training, THE Finetune_Evaluator SHALL save the fine-tuned model checkpoint to disk and evaluate on the ETTh1_Test_Set, computing MAE, MSE, MASE, and CRPS metrics.
5. WHEN the Finetune_Evaluator completes evaluation, THE Finetune_Evaluator SHALL measure and record the Inference_Time in seconds (rounded to 2 decimal places) for test set forecasting.
6. IF the pretrained checkpoint file does not exist at the specified path, THEN THE Finetune_Evaluator SHALL raise an error indicating the checkpoint is missing and abort without partial training.
7. IF a NaN loss value is detected during any training epoch, THEN THE Finetune_Evaluator SHALL halt training immediately and report the epoch number at which divergence occurred.

### Requirement 6: Metrics Implementation

**User Story:** As a researcher, I want clearly documented metric implementations, so that I can understand exactly how each model is being evaluated and reproduce the results.

#### Acceptance Criteria

1. THE Metrics_Module SHALL implement MAE (Mean Absolute Error) as the average of absolute differences between predictions and actual values, computed over all elements of equal-shape input arrays containing finite numeric values.
2. THE Metrics_Module SHALL implement MSE (Mean Squared Error) as the average of squared differences between predictions and actual values, computed over all elements of equal-shape input arrays containing finite numeric values, producing a value greater than or equal to zero.
3. THE Metrics_Module SHALL implement MASE (Mean Absolute Scaled Error) as MAE divided by the MAE of a seasonal naive baseline with a seasonal period of 24 time steps, where MASE less than 1 indicates the model outperforms the seasonal naive baseline.
4. IF the seasonal naive baseline MAE equals zero when computing MASE, THEN THE Metrics_Module SHALL return positive infinity.
5. THE Metrics_Module SHALL implement CRPS (Continuous Ranked Probability Score) approximated from quantile predictions by computing the pinball loss for each quantile level (P10, P50, P90), averaging across all levels, and multiplying by the factor (2 / number_of_quantile_levels).
6. THE Metrics_Module SHALL include inline code comments for each metric function that state the mathematical formula and a plain-language explanation of what the metric measures.
7. WHEN the same prediction and target arrays of equal shape containing finite numeric values are passed to any metric function multiple times, THE Metrics_Module SHALL produce identical results (deterministic computation).

### Requirement 7: Results Comparison Table

**User Story:** As a researcher, I want a clean formatted comparison table showing all 5 models side by side, so that I can quickly assess relative performance.

#### Acceptance Criteria

1. WHEN all model evaluations have completed and their metric dictionaries are available, THE Results_Table SHALL print a formatted table to stdout with columns: Model, MAE, MSE, MASE, CRPS, and Inference_Time.
2. THE Results_Table SHALL include rows for all 5 models in this order: Naive baseline, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned).
3. WHEN the Results_Table is generated, THE Results_Table SHALL save the complete metrics to `evaluation/results/final_metrics.json` as a JSON file with model names as top-level keys and metric dictionaries (containing mae, mse, mase, crps, inference_time) as values, with numeric values rounded to 4 decimal places.
4. THE Results_Table SHALL display CRPS as "N/A" for point-forecast-only baselines (Naive, ARIMA, Prophet) since they do not produce probabilistic outputs.
5. THE Results_Table SHALL display Inference_Time in seconds with appropriate units (e.g., "<1ms", "~Xs", "~Xms") for readability.

### Requirement 8: Forecast Visualization Plot

**User Story:** As a researcher, I want a publication-quality forecast plot showing actual values versus model predictions with uncertainty bands, so that I can visually assess forecast quality.

#### Acceptance Criteria

1. WHEN the Visualization_Module generates the forecast plot, THE Visualization_Module SHALL display actual values as a solid line and P50 predictions as a dashed line over the configured forecast horizon (96 time steps).
2. WHEN the Visualization_Module generates the forecast plot, THE Visualization_Module SHALL shade the region between P10 and P90 predictions with a semi-transparent fill (alpha between 0.2 and 0.4) to show the 80% prediction interval.
3. WHEN the Visualization_Module generates the forecast plot, THE Visualization_Module SHALL save the plot as a PNG file at minimum 300 DPI to the `evaluation/results/` directory, creating the directory if it does not exist.
4. THE Visualization_Module SHALL include an x-axis label indicating time steps, a y-axis label indicating value scale, a legend identifying all three visual elements (actual line, P50 prediction line, and P10-P90 interval), and a title containing the forecast window index and dataset name.
5. IF the actual values array and prediction arrays have different lengths, THEN THE Visualization_Module SHALL raise an error indicating the dimension mismatch without saving a plot file.
6. WHEN the Visualization_Module generates the forecast plot, THE Visualization_Module SHALL render the actual line and P50 prediction line in visually distinct colors with different line styles so they are distinguishable without color (solid vs dashed).

### Requirement 9: Pretraining Loss Curve Plot

**User Story:** As a researcher, I want a loss curve plot showing pretraining reconstruction loss across all 3 domains over epochs, so that I can verify training convergence.

#### Acceptance Criteria

1. WHEN the Visualization_Module generates the loss curve plot, THE Visualization_Module SHALL display reconstruction loss on the y-axis (labeled "Reconstruction Loss") and epoch number on the x-axis (labeled "Epoch"), with one data point per completed epoch for each domain.
2. WHEN the Visualization_Module generates the loss curve plot, THE Visualization_Module SHALL show separate lines for each of the 3 pretraining domains (Energy, Weather, Finance) with distinct colors and a legend identifying each domain by name.
3. WHEN the Visualization_Module generates the loss curve plot, THE Visualization_Module SHALL save the plot as a PNG file at minimum 300 DPI to `evaluation/results/` with the filename `pretraining_loss_curve.png`.
4. IF the per-domain loss data for any domain contains zero epochs of data, THEN THE Visualization_Module SHALL omit that domain's line from the plot and still render the remaining domains.

### Requirement 10: Zero-Shot Comparison Bar Chart

**User Story:** As a researcher, I want a bar chart comparing MAE across all 5 models, so that I can immediately see which models perform best.

#### Acceptance Criteria

1. WHEN the Visualization_Module generates the comparison bar chart, THE Visualization_Module SHALL display one bar per model in the following left-to-right order: Naive, ARIMA, Prophet, PatchTST zero-shot, PatchTST fine-tuned, with MAE values on the y-axis starting at zero.
2. WHEN the Visualization_Module generates the comparison bar chart, THE Visualization_Module SHALL use one color for baseline models (Naive, ARIMA, Prophet) and a different color for PatchTST models (zero-shot, fine-tuned) to visually separate classical and deep learning approaches, and include a legend identifying each color group.
3. WHEN the Visualization_Module generates the comparison bar chart, THE Visualization_Module SHALL include value labels on top of each bar showing the exact MAE value rounded to 4 decimal places.
4. WHEN the Visualization_Module generates the comparison bar chart, THE Visualization_Module SHALL include a chart title containing "MAE Comparison", label the x-axis with model names, and label the y-axis with "MAE".
5. WHEN the Visualization_Module generates the comparison bar chart, THE Visualization_Module SHALL save the plot as a PNG file at minimum 300 DPI to `evaluation/results/` with the filename `mae_comparison_bar_chart.png`.
