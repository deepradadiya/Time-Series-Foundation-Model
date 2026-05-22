# Requirements Document

## Introduction

This feature implements the complete data pipeline for the Time Series Foundation Model project. The pipeline downloads four real-world time series datasets (Energy, Weather, Finance, ETTh1) from their canonical sources, preprocesses them through normalization, patching, windowing, and chronological splitting, and saves the results as PyTorch tensors for efficient training. The pipeline targets Google Colab free tier constraints (T4 GPU, ~15GB VRAM) and respects the PatchTST architecture's input requirements (CONTEXT_LENGTH=512, PATCH_LEN=16, PATCH_STRIDE=8, FORECAST_HORIZON=96).

## Glossary

- **Pipeline**: The end-to-end system that downloads raw datasets, preprocesses them, and produces PyTorch tensors ready for model training.
- **Energy_Downloader**: The script (data/download_energy.py) responsible for downloading the UCI Electricity Load dataset.
- **Weather_Downloader**: The script (data/download_weather.py) responsible for downloading the Weather dataset.
- **Finance_Downloader**: The script (data/download_finance.py) responsible for downloading Bitcoin hourly OHLCV data.
- **ETTh1_Downloader**: The script (data/download_etth1.py) responsible for downloading the ETTh1 benchmark dataset.
- **Preprocessor**: The script (data/preprocess.py) responsible for normalization, patching, windowing, splitting, and saving processed data.
- **Verifier**: The script (data/verify_data.py) responsible for loading processed datasets and printing summary statistics.
- **Instance_Normalization**: Z-score normalization applied per time series: subtract mean, divide by standard deviation.
- **Patch**: A fixed-length window of consecutive time steps (PATCH_LEN=16) treated as a single input token for the transformer.
- **Stride**: The step size (PATCH_STRIDE=8) between the start of consecutive patches, creating 50% overlap.
- **Context_Window**: A sliding window of CONTEXT_LENGTH=512 time steps used as one training sample input.
- **Forecast_Horizon**: The number of future time steps (96) the model predicts after a context window.
- **Chronological_Split**: Splitting data by time order (train first, then val, then test) to prevent future data leakage.
- **Normalization_Stats**: The per-dataset mean and standard deviation values saved for denormalization at inference time.

## Requirements

### Requirement 1: Download Energy Dataset

**User Story:** As a researcher, I want to download the UCI Electricity Load dataset from HuggingFace datasets, so that I can use real-world electricity consumption patterns for pretraining.

#### Acceptance Criteria

1. WHEN the Energy_Downloader is executed, THE Energy_Downloader SHALL download the electricity hourly dataset from HuggingFace datasets ("monash_tsf/electricity_hourly") as the primary source, falling back to the Monash Time Series repository if the primary source is unavailable.
2. WHEN the download completes, THE Energy_Downloader SHALL extract a univariate series of 100,000 consecutive time steps by selecting the first household (index 0) from the 321-household hourly electricity consumption data, starting from the first available time step.
3. WHEN the extraction completes, THE Energy_Downloader SHALL save the result as data/raw/energy.csv with columns "timestamp" (ISO 8601 format) and "value" (float), containing exactly 100,000 rows.
4. WHEN the file is saved, THE Energy_Downloader SHALL print statistics including: series length, minimum value, maximum value, mean value, and number of NaN values, with numeric values displayed to 4 decimal places.
5. IF the download fails due to a network error, THEN THE Energy_Downloader SHALL retry with exponential backoff starting at 2 seconds (doubling each attempt) up to 3 attempts before raising an error.
6. IF the dataset file already exists at data/raw/energy.csv, THEN THE Energy_Downloader SHALL skip the download and print a message indicating the file already exists.
7. IF the selected household series contains fewer than 100,000 time steps, THEN THE Energy_Downloader SHALL raise an error indicating the available length and the required length of 100,000.

### Requirement 2: Download Weather Dataset

**User Story:** As a researcher, I want to download the Weather dataset used in the PatchTST paper, so that I can use meteorological time series for pretraining.

#### Acceptance Criteria

1. WHEN the Weather_Downloader is executed, THE Weather_Downloader SHALL download the WTH.csv file from the PatchTST repository (https://github.com/yuqinie98/PatchTST) using an HTTP GET request with a 30-second connection timeout.
2. WHEN the download completes, THE Weather_Downloader SHALL extract the "OT" (oil temperature) column as a univariate series and map the "date" column from the source file to the "timestamp" column in the output.
3. WHEN the extraction completes, THE Weather_Downloader SHALL save the result as data/raw/weather.csv with exactly two columns named "timestamp" and "value", where "timestamp" preserves the original date format from the source file and "value" contains the OT column values.
4. WHEN the file is saved, THE Weather_Downloader SHALL print statistics to standard output including: series length (number of rows), minimum value, maximum value, mean value (rounded to 4 decimal places), and number of NaN values.
5. IF the download fails due to a network error or timeout, THEN THE Weather_Downloader SHALL retry with exponential backoff starting at 2 seconds (doubling each attempt) up to a maximum of 3 total attempts, and raise an error with the last failure reason if all attempts are exhausted.
6. IF the dataset file already exists at data/raw/weather.csv, THEN THE Weather_Downloader SHALL skip the download and print a message indicating the file already exists.
7. IF the downloaded file does not contain an "OT" column, THEN THE Weather_Downloader SHALL raise an error indicating the expected column is missing from the source file.
8. IF the downloaded file contains fewer than 1000 rows, THEN THE Weather_Downloader SHALL delete the file and raise an error indicating the download is corrupt or incomplete.

### Requirement 3: Download Finance Dataset

**User Story:** As a researcher, I want to download Bitcoin hourly price data, so that I can use financial time series for pretraining.

#### Acceptance Criteria

1. WHEN the Finance_Downloader is executed, THE Finance_Downloader SHALL download Bitcoin hourly OHLCV data using the yfinance library with ticker "BTC-USD", interval "1h", and period "2y".
2. WHEN the download completes, THE Finance_Downloader SHALL extract only the "Close" price column as a univariate series.
3. WHEN the extraction completes, THE Finance_Downloader SHALL save the result as data/raw/finance.csv with columns "timestamp" (ISO 8601 format, e.g. "2023-01-15 08:00:00") and "value" (Close price as a decimal number).
4. WHEN the file is saved, THE Finance_Downloader SHALL print statistics including: series length (number of rows), minimum value (rounded to 2 decimal places), maximum value (rounded to 2 decimal places), mean value (rounded to 2 decimal places), and number of NaN values.
5. IF the download fails due to a network error, THEN THE Finance_Downloader SHALL retry with exponential backoff starting at 2 seconds (doubling each retry) for a maximum of 3 total attempts before raising an error indicating the download failed.
6. IF the dataset file already exists at data/raw/finance.csv, THEN THE Finance_Downloader SHALL skip the download and print a message indicating the file already exists.
7. IF the downloaded data contains fewer than 1000 rows, THEN THE Finance_Downloader SHALL raise an error indicating insufficient data was returned.
8. WHEN the extraction completes and NaN values are present in the Close price column, THE Finance_Downloader SHALL forward-fill NaN values before saving, and report the original NaN count in the printed statistics.

### Requirement 4: Download ETTh1 Dataset

**User Story:** As a researcher, I want to download the ETTh1 benchmark dataset, so that I can use it exclusively for zero-shot evaluation of the pretrained model.

#### Acceptance Criteria

1. WHEN the ETTh1_Downloader is executed, THE ETTh1_Downloader SHALL download ETTh1.csv from https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv.
2. WHEN the download completes, THE ETTh1_Downloader SHALL extract only the "OT" (oil temperature) column as a univariate series and rename the source "date" column to "timestamp".
3. WHEN the extraction completes, THE ETTh1_Downloader SHALL save the result as data/raw/etth1.csv with exactly two columns: "timestamp" (preserving the original datetime strings from the source) and "value" (containing the OT values as floating-point numbers).
4. THE ETTh1_Downloader SHALL include a warning comment at the top of the source file, on its own line, containing the text "WARNING: This dataset must NEVER be used during pretraining. It is only for zero-shot evaluation."
5. IF the download fails due to a network error, THEN THE ETTh1_Downloader SHALL retry with exponential backoff starting at 2 seconds and doubling each attempt, up to a maximum of 3 attempts before raising an error.
6. IF the dataset file already exists at data/raw/etth1.csv, THEN THE ETTh1_Downloader SHALL skip the download and print a message indicating the file already exists.
7. IF the downloaded CSV does not contain a column named "OT", THEN THE ETTh1_Downloader SHALL raise an error indicating the expected column is missing and delete the downloaded file.
8. WHEN the extraction completes, THE ETTh1_Downloader SHALL verify that the saved file contains at least 17,000 rows.

### Requirement 5: Instance Normalization

**User Story:** As a researcher, I want each time series normalized to zero mean and unit variance, so that datasets with different scales (kWh vs Celsius vs USD) can be learned jointly by the model.

#### Acceptance Criteria

1. WHEN the Preprocessor normalizes a dataset, THE Preprocessor SHALL apply per-channel z-score normalization (subtract training split mean, divide by training split standard deviation) to each of the training, validation, and test splits using statistics computed exclusively from the training split.
2. THE Preprocessor SHALL compute normalization statistics exclusively from the training split to prevent data leakage into validation and test sets.
3. WHEN normalization is complete, THE Preprocessor SHALL save the per-dataset mean and standard deviation as a JSON file named `{dataset_name}_norm_stats.json` in data/processed/, containing a dictionary with "mean" and "std" keys each mapping to a list of per-channel float values.
4. IF a channel in the training split has zero standard deviation, THEN THE Preprocessor SHALL set that channel's standard deviation to 1.0 and emit a warning identifying the affected channel index.
5. WHEN inverse normalization is applied to previously normalized data using the same statistics, THE Preprocessor SHALL produce values matching the original input within an absolute tolerance of 1e-10.
6. THE Preprocessor SHALL accept both 1D arrays (univariate, single channel) and 2D arrays of shape (time_steps, num_channels) as input, treating 1D input as a single-channel series and returning output in the same dimensionality as the input.

### Requirement 6: Patching

**User Story:** As a researcher, I want time series segmented into fixed-length overlapping patches, so that the PatchTST transformer can process them as token sequences with reduced sequence length.

#### Acceptance Criteria

1. WHEN the Preprocessor creates patches from a 1-D normalized series of length L where L >= PATCH_LEN, THE Preprocessor SHALL segment the series into patches of length PATCH_LEN=16 with stride PATCH_STRIDE=8, producing floor((L - PATCH_LEN) / PATCH_STRIDE) + 1 patches each containing PATCH_LEN consecutive time steps.
2. WHEN a context window of 512 time steps is patched, THE Preprocessor SHALL produce exactly 63 patches of shape (63, 16) with 50% overlap between consecutive patches.
3. THE Preprocessor SHALL discard trailing time steps that cannot form a complete patch of length PATCH_LEN, including only patches whose start index plus PATCH_LEN does not exceed the series length.
4. IF the input series length is less than PATCH_LEN, THEN THE Preprocessor SHALL raise a ValueError indicating that the series is too short to form at least one complete patch.
5. IF the input series is not 1-D, THEN THE Preprocessor SHALL raise a ValueError indicating that each channel must be processed independently as a 1-D array.
6. THE Preprocessor SHALL preserve the numeric data type of the input series in the output patch array.

### Requirement 7: Windowing

**User Story:** As a researcher, I want sliding windows extracted from the time series, so that each window becomes one training sample with a corresponding forecast target.

#### Acceptance Criteria

1. WHEN the Preprocessor creates training samples from a time series, THE Preprocessor SHALL extract sliding windows of length CONTEXT_LENGTH=512 time steps with a step size of 96 time steps, treating each channel independently for multivariate series.
2. WHEN a context window is extracted, THE Preprocessor SHALL assign the next FORECAST_HORIZON=96 time steps immediately following the window as the forecast target (label), producing a context tensor of shape (512,) and a target tensor of shape (96,) both as float32 values.
3. THE Preprocessor SHALL discard any trailing portion of the series that cannot form a complete context window plus forecast horizon (512 + 96 = 608 time steps minimum).
4. IF a series has fewer than 608 time steps in total, THEN THE Preprocessor SHALL generate zero samples from that series and log a warning identifying the discarded series.

### Requirement 8: Chronological Splitting

**User Story:** As a researcher, I want data split by time order rather than randomly, so that the model never trains on future data that would leak information about the test period.

#### Acceptance Criteria

1. WHEN the Preprocessor splits a dataset, THE Preprocessor SHALL allocate the first 70% of time steps to training, the next 15% to validation, and the final 15% to testing, where split boundaries are computed by truncating fractional indices toward zero.
2. THE Preprocessor SHALL perform the split along the time axis without any shuffling or random sampling.
3. THE Preprocessor SHALL apply windowing independently within each split so that no window of size context length plus forecast horizon spans across split boundaries, discarding any window position that would require time steps from an adjacent split.
4. WHEN the Preprocessor computes normalization statistics, THE Preprocessor SHALL derive mean and standard deviation exclusively from the training split and apply those statistics to normalize all three splits.
5. WHILE the ETTh1 dataset is being processed, THE Preprocessor SHALL mark the ETTh1 processed output as zero-shot evaluation only and exclude it from the pretraining MultiDomainDataLoader domain list.
6. IF a split contains fewer time steps than one full window of size context length plus forecast horizon, THEN THE Preprocessor SHALL log a warning and produce zero samples for that split.

### Requirement 9: Save as PyTorch Tensors

**User Story:** As a researcher, I want processed data saved as PyTorch tensors, so that training can load data quickly without repeating preprocessing.

#### Acceptance Criteria

1. WHEN preprocessing completes for a dataset, THE Preprocessor SHALL save the train, validation, and test splits as PyTorch .pt tensor files in data/processed/ with filenames following the pattern {dataset_name}_{split}.pt (e.g., "etth1_train.pt", "etth1_val.pt", "etth1_test.pt").
2. THE Preprocessor SHALL generate samples from each split using a sliding window of size CONTEXT_LENGTH + FORECAST_HORIZON (512 + 96 = 608 time steps) with a stride of FORECAST_HORIZON (96), and save each split as a dictionary containing "context" tensors of shape (num_samples, 512) and "target" tensors of shape (num_samples, 96).
3. WHEN saving tensors, THE Preprocessor SHALL use float32 dtype for all tensor values.
4. IF a split contains fewer than CONTEXT_LENGTH + FORECAST_HORIZON (608) time steps, THEN THE Preprocessor SHALL save that split as a dictionary with empty tensors of shape (0, 512) for "context" and (0, 96) for "target", and log a warning indicating insufficient data for that split.

### Requirement 10: Data Verification

**User Story:** As a researcher, I want to verify all processed datasets at a glance, so that I can confirm the pipeline ran correctly before starting training.

#### Acceptance Criteria

1. WHEN the Verifier is executed, THE Verifier SHALL load all 4 processed datasets (energy, weather, finance, etth1) from data/processed/ and print a summary table with columns: Dataset, Train samples, Val samples, Test samples, Num patches, Value range (minimum and maximum values across all splits per dataset).
2. WHEN displaying ETTh1 statistics in the summary table, THE Verifier SHALL annotate the Num patches column with "[ZERO-SHOT ONLY]" to indicate this dataset is reserved for evaluation.
3. WHEN the summary table is printed, THE Verifier SHALL plot one sample time series per dataset by selecting the first context window (512 time steps) from the test split of each dataset and rendering it as a line chart.
4. IF any processed dataset file is missing, THEN THE Verifier SHALL print an error message identifying the missing dataset by name, skip that dataset's row in the summary table and visualization, and continue verifying the remaining datasets.
