# Implementation Plan: Gradio Demo & HuggingFace Publish

## Overview

This plan implements the multi-tab Gradio application with Plotly charts, the HuggingFace Hub publishing script, and the final README generator. Tasks are ordered to build foundational components first (frequency detection, inference pipeline, chart builders), then assemble them into the Gradio tabs, followed by the publishing and README modules, and finally wire everything together.

## Tasks

- [x] 1. Implement core utility components
  - [x] 1.1 Implement FrequencyDetector in `app/gradio_app.py`
    - Create `detect_frequency(timestamps: pd.Series) -> tuple[str, str | None]` function
    - Compute median interval between consecutive timestamps
    - Classify using tolerance bands: hourly [30min, 90min], daily [12h, 36h], weekly [5d, 9d]
    - Default to "daily" with warning if interval falls outside all bands
    - _Requirements: 1.1, 1.9_

  - [x]* 1.2 Write property test for FrequencyDetector
    - **Property 1: Frequency detection correctness**
    - Generate random timestamp arrays with known frequencies + jitter within tolerance bands
    - Verify correct classification for hourly, daily, weekly
    - Verify default-to-daily with warning for out-of-band intervals
    - **Validates: Requirements 1.1, 1.9**

  - [x] 1.3 Implement ModelCache and InferencePipeline in `app/gradio_app.py`
    - Create `ModelCache` class with singleton pattern for model/head caching
    - Implement `get_model()` classmethod with 60-second timeout, fallback to random weights
    - Implement `run_forecast(series, horizon, model, head) -> np.ndarray` returning shape (horizon, 3)
    - Handle custom horizon heads for non-96 horizons
    - Normalize using z-score from last 512 steps, inverse-normalize output
    - _Requirements: 1.3, 6.3, 6.4, 6.5, 6.6_

  - [x]* 1.4 Write property test for inference output validity
    - **Property 2: Inference output validity**
    - Generate random float arrays of length 512-2048, horizons from {24, 48, 96, 192}
    - Verify output shape is (horizon, 3) and P10[t] <= P50[t] <= P90[t] for all t
    - **Validates: Requirements 1.3**

  - [x]* 1.5 Write property test for model caching idempotence
    - **Property 11: Model caching idempotence**
    - Call ModelCache.get_model() N times (N >= 2)
    - Verify all calls return identical Python object (same id())
    - Verify model loaded from disk at most once
    - **Validates: Requirements 6.3**

- [x] 2. Implement Plotly chart builders and metrics logic
  - [x] 2.1 Implement PlotlyChartBuilder in `app/gradio_app.py`
    - Create `build_forecast_chart()` with traces: blue solid historical, orange dashed P50, light orange shaded P10-P90 band, optional green dotted actuals
    - Create `build_benchmark_chart()` with traces: ground truth, ARIMA, Prophet, PatchTST P50 + P10-P90 band
    - Add MAE values in legend entries; append ★ to best method
    - Enable interactive zoom, pan, and hover tooltips
    - _Requirements: 1.4, 2.2, 2.3, 6.2_

  - [x] 2.2 Implement metrics availability detection logic
    - Compute MAE and MASE when CSV length >= 512 + horizon (actuals available)
    - Return None when CSV length < 512 + horizon
    - Use seasonal period of 24 for MASE computation
    - _Requirements: 1.5_

  - [x]* 2.3 Write property test for metrics availability detection
    - **Property 3: Metrics availability detection**
    - Generate random arrays of length 512-1024, random horizons
    - Verify metrics computed (finite positive floats) when length >= 512 + horizon
    - Verify metrics are None when length < 512 + horizon
    - **Validates: Requirements 1.5**

  - [x]* 2.4 Write property test for best method identification
    - **Property 4: Best method identification**
    - Generate random dicts of 2-5 methods with random positive MAE floats
    - Verify exactly one method gets the best-indicator label
    - Verify it is the method with strictly lowest MAE
    - **Validates: Requirements 2.3**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement BenchmarkCache and Gradio tab assembly
  - [x] 4.1 Implement BenchmarkCache in `app/gradio_app.py`
    - Create `BenchmarkCache` class storing 10 pre-computed ETTh1 test samples
    - Load from `app/benchmark_cache.npz` at startup; compute and save if missing
    - Each sample: start_index, ground_truth, arima_forecast, prophet_forecast, patchtst_p10/p50/p90, mae scores
    - Use `evaluation/baselines.py` for ARIMA/Prophet computation
    - _Requirements: 2.1, 2.4_

  - [x] 4.2 Assemble Upload Tab (Tab 1) in `app/gradio_app.py`
    - File upload widget (CSV, max 50MB)
    - Radio buttons or dropdown for horizon selection: [24, 48, 96, 192]
    - Target column dropdown (populated on upload)
    - Forecast button triggering inference pipeline
    - Plotly chart output and status textbox
    - CSV validation with error messages for < 512 rows, no numeric cols, no datetime col
    - Display warning banner if model loaded with random weights
    - _Requirements: 1.1-1.9, 6.1, 6.4, 6.5, 6.6_

  - [x] 4.3 Assemble Benchmark Tab (Tab 2) in `app/gradio_app.py`
    - Dropdown with 10 ETTh1 samples (identified by start index)
    - On selection: render benchmark comparison chart from cache
    - Chart renders within 2 seconds (pre-computed data)
    - _Requirements: 2.1-2.4_

  - [x] 4.4 Assemble About Tab (Tab 3) in `app/gradio_app.py`
    - ASCII architecture diagram in fixed-width text box
    - Three pretraining domains with one-sentence descriptions
    - Metrics table (MAE, MSE, MASE, CRPS for PatchTST zero-shot, fine-tuned, ARIMA, Prophet)
    - Clickable links to HuggingFace Hub and GitHub (open in new tab)
    - _Requirements: 3.1-3.4_

  - [x] 4.5 Wire Gradio Blocks app with three tabs and entry point
    - Create `create_app() -> gr.Blocks` with tabs: "Upload your own data", "Live benchmark demo", "About the model"
    - Set "Upload your own data" as default active tab
    - Add `if __name__ == "__main__"` entry point launching on 0.0.0.0:7860
    - _Requirements: 6.1_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement HuggingFace Hub publishing script
  - [x] 6.1 Implement retry_with_backoff in `publish_to_hub.py`
    - Create `retry_with_backoff(func, max_retries=3, base_delay=2.0)` decorator/function
    - Exponential backoff: 2s, 4s, 8s delays between retries
    - Raise last exception if all retries exhausted
    - _Requirements: 4.6_

  - [x]* 6.2 Write property test for retry with exponential backoff
    - **Property 6: Retry with exponential backoff**
    - Mock functions that fail N times (0 <= N <= 3) before succeeding
    - Verify exactly min(N+1, 4) calls total
    - Verify exception raised when N >= 3
    - Verify delay pattern: base_delay * 2^k
    - **Validates: Requirements 4.6**

  - [x] 6.3 Implement ModelCardGenerator in `publish_to_hub.py`
    - Create `generate_model_card(model_name, param_count, domains, metrics, github_url) -> str`
    - Include sections: model name, architecture summary with param count, domains with row counts, benchmark table, 5-line Python usage example, GitHub URL
    - _Requirements: 4.4_

  - [x]* 6.4 Write property test for model card completeness
    - **Property 5: Model card completeness**
    - Generate random valid inputs (model name, param count, domain dict, metrics dict, URL)
    - Verify all required sections present in output
    - **Validates: Requirements 4.4**

  - [x] 6.5 Implement publish_all orchestrator in `publish_to_hub.py`
    - Validate HF_TOKEN environment variable (exit 1 if missing)
    - Validate checkpoint files exist (exit 1 if missing)
    - Resolve username via HF Hub API whoami
    - Push pretrained checkpoint + model card to `{username}/patchtst-foundation-pretrained`
    - Push fine-tuned checkpoint + model card to `{username}/patchtst-etth1-finetuned`
    - Deploy Gradio Space to `{username}/timeseries-foundation-demo`
    - Use retry_with_backoff for all network operations
    - _Requirements: 4.1-4.3, 4.5, 4.7, 4.8_

- [x] 7. Implement README generator
  - [x] 7.1 Implement "What This Is" section generator
    - Produce exactly 3 sentences, each <= 25 words
    - No domain-specific acronyms without inline definitions
    - _Requirements: 5.1_

  - [x]* 7.2 Write property test for README "What This Is" constraints
    - **Property 7: README "What This Is" constraints**
    - Generate random metric inputs
    - Verify exactly 3 sentences, each <= 25 words, no undefined acronyms
    - **Validates: Requirements 5.1**

  - [x] 7.3 Implement results table formatter
    - Generate markdown table with columns MAE, MSE, MASE, CRPS
    - Rows in order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned)
    - Round all values to 4 decimal places
    - Display "N/A" for CRPS on Naive, ARIMA, Prophet
    - _Requirements: 5.3_

  - [x]* 7.4 Write property test for results table formatting
    - **Property 8: Results table formatting**
    - Generate random metric dicts with 5 models
    - Verify column order, row order, 4 decimal places, N/A for point-forecast CRPS
    - **Validates: Requirements 5.3**

  - [x] 7.5 Implement "How to Reproduce" section generator
    - Generate exactly 8 numbered commands covering all pipeline stages
    - Each command starts with valid shell prefix (pip, python, bash) or Python module invocation
    - Commands executable in Google Colab
    - _Requirements: 5.5_

  - [x]* 7.6 Write property test for README reproduction steps count
    - **Property 9: README reproduction steps count**
    - Verify exactly 8 numbered commands
    - Verify each starts with valid shell command prefix
    - **Validates: Requirements 5.5**

  - [x] 7.7 Implement resume bullet generator
    - Include PatchTST zero-shot MAE (4 decimal places) and at least one baseline MAE
    - Express as comparative statement
    - _Requirements: 5.6_

  - [x]* 7.8 Write property test for resume bullet metric inclusion
    - **Property 10: Resume bullet metric inclusion**
    - Generate random metric dicts with PatchTST and baseline MAE values
    - Verify PatchTST MAE and at least one baseline MAE appear in output
    - **Validates: Requirements 5.6**

  - [x] 7.9 Assemble full generate_readme function
    - Combine all sections: What This Is, Architecture Diagram, Results Table, Pretraining Details, How to Reproduce, Resume Bullet, Links (4 URLs)
    - Read metrics from `evaluation/results/final_metrics.json`
    - Read config from `config.py`
    - Write output to `README.md`
    - _Requirements: 5.1-5.7_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Integration and final wiring
  - [x] 9.1 Create benchmark cache generation script
    - Add logic to generate `app/benchmark_cache.npz` from ETTh1 test data
    - Use evaluation/baselines.py for ARIMA/Prophet forecasts
    - Use model inference for PatchTST forecasts
    - Compute MAE for each method per sample
    - Can be run standalone or triggered by BenchmarkCache on first access
    - _Requirements: 2.1, 2.4_

  - [x] 9.2 Wire publish script with README generation
    - Ensure `publish_to_hub.py` can invoke README generation before publishing
    - Add `requirements.txt` generation for Space deployment
    - Verify all file paths and imports resolve correctly
    - _Requirements: 4.3, 5.1-5.7_

  - [x]* 9.3 Write integration tests for end-to-end flows
    - Test CSV upload → frequency detection → forecast → chart rendering
    - Test benchmark sample selection → chart rendering
    - Test publish script with mocked HF Hub API
    - _Requirements: 1.1-1.9, 2.1-2.4, 4.1-4.8_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The existing `app/app.py` is preserved; the new app lives at `app/gradio_app.py`
- All 11 correctness properties from the design are covered by property test tasks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.5", "2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "6.1"] },
    { "id": 4, "tasks": ["4.5", "6.2", "6.3", "7.1"] },
    { "id": 5, "tasks": ["6.4", "6.5", "7.2", "7.3", "7.5", "7.7"] },
    { "id": 6, "tasks": ["7.4", "7.6", "7.8", "7.9"] },
    { "id": 7, "tasks": ["9.1", "9.2"] },
    { "id": 8, "tasks": ["9.3"] }
  ]
}
```
