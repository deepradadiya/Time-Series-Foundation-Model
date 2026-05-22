# Implementation Plan: Data Pipeline

## Overview

This plan implements the complete data pipeline for the Time Series Foundation Model. The pipeline consists of four domain-specific download scripts, a unified preprocessing pipeline that leverages existing modules (`data/preprocess.py`, `data/patching.py`), and a verification script. Implementation follows the dependency order: download scripts (parallel) → preprocessing pipeline → verification → tests.

## Tasks

- [x] 1. Implement download scripts
  - [x] 1.1 Create `data/download_energy.py`
    - Implement `download_energy()` function that downloads from HuggingFace `monash_tsf/electricity_hourly`
    - Extract first household (index 0), first 100,000 time steps
    - Save as `data/raw/energy.csv` with "timestamp" (ISO 8601) and "value" columns
    - Implement retry with exponential backoff (2s, 4s, 8s) up to 3 attempts using `Config.MAX_RETRIES` and `Config.RETRY_BASE_DELAY`
    - Skip download if file already exists, print statistics on completion (length, min, max, mean to 4 decimal places, NaN count)
    - Raise `ValueError` if series has fewer than 100,000 time steps
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.2 Create `data/download_weather.py`
    - Implement `download_weather()` function that downloads WTH.csv from PatchTST GitHub repository
    - Extract "OT" (oil temperature) column, map "date" to "timestamp"
    - Save as `data/raw/weather.csv` with "timestamp" and "value" columns
    - Implement retry with exponential backoff up to 3 attempts with 30-second connection timeout
    - Skip download if file already exists, print statistics on completion
    - Raise `ValueError` if "OT" column missing; delete file and raise error if fewer than 1000 rows
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x] 1.3 Create `data/download_finance.py`
    - Implement `download_finance()` function using yfinance with ticker "BTC-USD", interval "1h", period "2y"
    - Extract "Close" column, forward-fill NaN values before saving
    - Save as `data/raw/finance.csv` with "timestamp" (ISO 8601) and "value" columns
    - Implement retry with exponential backoff up to 3 attempts
    - Skip download if file already exists, print statistics (2 decimal places for prices, report original NaN count)
    - Raise `ValueError` if fewer than 1000 rows returned
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 1.4 Create `data/download_etth1.py`
    - Implement `download_etth1()` function that downloads ETTh1.csv from `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`
    - Extract "OT" column, rename "date" to "timestamp"
    - Save as `data/raw/etth1.csv` with "timestamp" and "value" columns
    - Include warning comment in source file: "WARNING: This dataset must NEVER be used during pretraining. It is only for zero-shot evaluation."
    - Implement retry with exponential backoff up to 3 attempts
    - Skip download if file already exists; raise error and delete file if "OT" column missing
    - Verify saved file contains at least 17,000 rows
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [x] 2. Checkpoint - Verify download scripts
  - Ensure all download scripts can be imported without errors, ask the user if questions arise.

- [x] 3. Implement preprocessing pipeline
  - [x] 3.1 Create `data/preprocess_pipeline.py`
    - Implement `create_windows(data, context_length=512, forecast_horizon=96, stride=96)` that extracts sliding windows returning `(contexts, targets)` arrays
    - Implement `process_dataset(dataset_name, raw_path, output_dir="data/processed")` that:
      1. Loads CSV and extracts "value" column as numpy array
      2. Calls `preprocess.split_chronological()` for 70/15/15 split
      3. Calls `preprocess.compute_normalization_stats()` from train split only
      4. Calls `preprocess.normalize()` on all three splits
      5. Calls `create_windows()` on each normalized split
      6. Saves each split as PyTorch `.pt` file with `{"context": tensor(N, 512), "target": tensor(N, 96)}` in float32
      7. Calls `preprocess.save_normalization_stats()` to persist stats as JSON
    - Implement `run_pipeline(datasets=None)` that processes all four datasets (defaults to `['energy', 'weather', 'finance', 'etth1']`)
    - Handle edge cases: series shorter than 608 steps (save empty tensors shape `(0, 512)` / `(0, 96)`, log warning)
    - Use file naming pattern: `{dataset_name}_{split}.pt`
    - Import and reuse `data.preprocess` functions directly
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 8.6, 9.1, 9.2, 9.3, 9.4_

  - [ ]* 3.2 Write property test for window extraction (Property 6: Window Count and Dimensions)
    - **Property 6: Window Count and Dimensions**
    - Use Hypothesis to generate 1-D float arrays of length >= 608
    - Verify `create_windows` produces exactly `(L - 608) // 96 + 1` context arrays of length 512 and same number of target arrays of length 96
    - Place in `tests/properties/test_windowing_properties.py`
    - **Validates: Requirements 7.1, 7.3**

  - [ ]* 3.3 Write property test for context-target adjacency (Property 7: Context and Target Adjacency)
    - **Property 7: Context and Target Adjacency**
    - Use Hypothesis to generate 1-D float arrays of length >= 608
    - Verify for every (context, target) pair that target contains values immediately following context in the original series
    - Place in `tests/properties/test_windowing_properties.py`
    - **Validates: Requirements 7.2**

  - [ ]* 3.4 Write property test for saved tensor format (Property 10: Saved Tensors Have Correct Shape and Dtype)
    - **Property 10: Saved Tensors Have Correct Shape and Dtype**
    - Use Hypothesis to generate arrays, run through `process_dataset`, load saved `.pt` files
    - Verify "context" tensor shape is `(num_samples, 512)`, "target" shape is `(num_samples, 96)`, both dtype float32
    - Place in `tests/properties/test_windowing_properties.py`
    - **Validates: Requirements 9.2, 9.3**

- [x] 4. Checkpoint - Verify preprocessing pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement verification script
  - [x] 5.1 Create `data/verify_data.py`
    - Implement `load_processed_dataset(dataset_name, processed_dir="data/processed")` that loads all three split `.pt` files
    - Implement `compute_dataset_stats(dataset_name, splits)` returning train/val/test sample counts, num_patches (using `patching.compute_num_patches`), value min/max
    - Implement `print_summary_table(all_stats)` that prints formatted table with columns: Dataset, Train samples, Val samples, Test samples, Num patches, Value range
    - Annotate ETTh1 Num patches column with "[ZERO-SHOT ONLY]"
    - Implement `plot_sample_series(all_splits, output_path=None)` that plots first test context window (512 steps) from each dataset as line charts
    - Implement `verify_all(processed_dir="data/processed")` as main entry point
    - Handle missing/corrupt `.pt` files gracefully: print error, skip dataset, continue
    - Add `if __name__ == "__main__"` entry point
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 5.2 Write unit tests for verification script
    - Test `load_processed_dataset` with mock `.pt` files
    - Test `compute_dataset_stats` returns correct structure
    - Test `print_summary_table` output format (capture stdout)
    - Test graceful degradation when files are missing
    - Place in `tests/unit/test_verify_data.py`
    - _Requirements: 10.1, 10.4_

- [ ] 6. Implement property-based tests for normalization and splitting
  - [ ]* 6.1 Write property test for normalization round trip (Property 1: Normalization Round Trip)
    - **Property 1: Normalization Round Trip**
    - Use Hypothesis to generate numpy arrays (1-D and 2-D) with finite float values
    - Verify `normalize` → `inverse_normalize` produces values within 1e-10 of original
    - Place in `tests/properties/test_normalization_properties.py`
    - **Validates: Requirements 5.5, 5.6**

  - [ ]* 6.2 Write property test for training split statistics (Property 2: Training Split Zero Mean and Unit Variance)
    - **Property 2: Training Split Zero Mean and Unit Variance**
    - Use Hypothesis to generate arrays of shape (time_steps, num_channels) with time_steps >= 10 and non-zero variance
    - Verify after split and normalize, training split has per-channel mean within 1e-10 of zero and std within 1e-10 of one
    - Place in `tests/properties/test_normalization_properties.py`
    - **Validates: Requirements 5.1, 5.2, 8.4**

  - [ ]* 6.3 Write property test for forward-fill (Property 3: Forward-Fill Removes NaN While Preserving Values)
    - **Property 3: Forward-Fill Removes NaN While Preserving Values**
    - Use Hypothesis to generate 1-D float arrays with NaN at arbitrary positions (non-NaN first element)
    - Verify forward-fill produces zero NaN values and non-NaN positions retain original values
    - Place in `tests/properties/test_normalization_properties.py`
    - **Validates: Requirements 3.8**

  - [ ]* 6.4 Write property test for chronological split preservation (Property 8: Chronological Split Preserves Data)
    - **Property 8: Chronological Split Preserves Data**
    - Use Hypothesis to generate numpy arrays
    - Verify concatenating train + val + test splits equals original input exactly
    - Place in `tests/properties/test_splitting_properties.py`
    - **Validates: Requirements 8.2**

  - [ ]* 6.5 Write property test for split size formula (Property 9: Split Sizes Follow Ratio Formula)
    - **Property 9: Split Sizes Follow Ratio Formula**
    - Use Hypothesis to generate arrays of length N >= 3
    - Verify train length = `int(N * 0.70)`, val length = `int(N * 0.85) - int(N * 0.70)`, test = remainder
    - Place in `tests/properties/test_splitting_properties.py`
    - **Validates: Requirements 8.1**

- [ ] 7. Implement property-based tests for patching
  - [ ]* 7.1 Write property test for patch count and dimensions (Property 4: Patch Count and Dimensions)
    - **Property 4: Patch Count and Dimensions**
    - Use Hypothesis to generate 1-D arrays of length L >= 16
    - Verify `create_patches` produces exactly `floor((L - 16) / 8) + 1` patches each of length 16
    - Verify each patch contains consecutive values from original series starting at index `i * 8`
    - Place in `tests/properties/test_patching_properties.py`
    - **Validates: Requirements 6.1, 6.3**

  - [ ]* 7.2 Write property test for patching dtype preservation (Property 5: Patching Preserves Data Type)
    - **Property 5: Patching Preserves Data Type**
    - Use Hypothesis to generate 1-D arrays of length >= 16 with dtype in {float32, float64, int32, int64}
    - Verify output array has same dtype as input
    - Place in `tests/properties/test_patching_properties.py`
    - **Validates: Requirements 6.6**

- [ ] 8. Implement unit tests for download scripts
  - [ ]* 8.1 Write unit tests for download scripts
    - Mock network calls (patch `urllib.request.urlopen`, `datasets.load_dataset`, `yfinance.download`)
    - Test retry logic triggers on network errors with correct backoff timing
    - Test skip behavior when file already exists
    - Test error conditions: missing columns, insufficient rows, insufficient time steps
    - Test correct CSV output format (two columns: timestamp, value)
    - Place in `tests/unit/test_download_energy.py`, `tests/unit/test_download_weather.py`, `tests/unit/test_download_finance.py`, `tests/unit/test_download_etth1.py`
    - _Requirements: 1.5, 1.6, 1.7, 2.5, 2.6, 2.7, 2.8, 3.5, 3.6, 3.7, 3.8, 4.5, 4.6, 4.7, 4.8_

  - [ ]* 8.2 Write unit tests for preprocessing pipeline
    - Test `create_windows` with known arrays and verify output shapes and values
    - Test `process_dataset` with synthetic CSV data end-to-end
    - Test edge case: series shorter than 608 produces empty tensors
    - Test normalization stats are saved correctly as JSON
    - Place in `tests/unit/test_preprocess_pipeline.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 9.1, 9.2, 9.3, 9.4_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases using pytest
- Download scripts (1.1–1.4) are independent and can be implemented in parallel
- The preprocessing pipeline (3.1) depends on download scripts existing (for imports/interface alignment)
- The verification script (5.1) depends on the preprocessing pipeline
- All download scripts reuse the retry pattern from `Config.MAX_RETRIES` and `Config.RETRY_BASE_DELAY`
- Existing modules (`data/preprocess.py`, `data/patching.py`, `data/dataset.py`) are reused as-is

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["3.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "3.4", "5.1"] },
    { "id": 3, "tasks": ["5.2", "6.1", "6.2", "6.3", "6.4", "6.5"] },
    { "id": 4, "tasks": ["7.1", "7.2", "8.1", "8.2"] }
  ]
}
```
