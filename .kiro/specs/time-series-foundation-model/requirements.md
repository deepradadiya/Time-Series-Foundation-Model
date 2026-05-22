# Requirements Document

## Introduction

This document specifies the requirements for a Time Series Foundation Model built from scratch. The model follows the PatchTST architecture and is pretrained on three real-world domains (Energy, Weather, Finance) using Masked Patch Modeling. The primary goal is zero-shot forecasting on an unseen benchmark dataset (ETTh1) that outperforms classical baselines (ARIMA, Prophet) without any fine-tuning. The model produces probabilistic forecasts (P10/P50/P90 prediction intervals) and is deployed as an interactive HuggingFace Space demo.

## Glossary

- **PatchTST_Model**: The PatchTST-style transformer model (6 layers, 256 hidden dim, 8 attention heads) that processes time series data as patches
- **Patch**: A fixed-length segment of a time series (length 16, stride 8) used as the atomic input unit to the transformer
- **Masked_Patch_Modeling**: A self-supervised pretraining objective where random patches are masked and the model learns to reconstruct them
- **Probabilistic_Head**: The output layer that produces quantile forecasts (P10, P50, P90) instead of a single point prediction
- **Zero_Shot_Forecasting**: Evaluating the pretrained model on a target dataset without any fine-tuning on that dataset
- **Context_Window**: The input sequence length (512 time steps) the model uses to generate forecasts
- **Forecast_Horizon**: The number of future time steps (96) the model predicts
- **ETTh1_Dataset**: The Electricity Transformer Temperature hourly benchmark dataset used exclusively for zero-shot evaluation
- **Energy_Dataset**: A multivariate electricity/energy consumption dataset used for pretraining
- **Weather_Dataset**: A multivariate weather observations dataset used for pretraining
- **Finance_Dataset**: A stock/crypto OHLCV (Open, High, Low, Close, Volume) dataset used for pretraining
- **CRPS**: Continuous Ranked Probability Score — a metric for evaluating probabilistic forecasts
- **Checkpoint**: A saved snapshot of model weights and optimizer state to Google Drive for recovery
- **Colab_Environment**: Google Colab free tier with T4 GPU, approximately 15GB VRAM, and sessions that reset every 12 hours
- **Gradio_App**: The interactive web application deployed on HuggingFace Spaces for forecasting demos
- **Data_Pipeline**: The preprocessing module that normalizes data, creates patches, and splits into train/val/test sets
- **Pretraining_Loop**: The training loop that iterates over all three domain datasets with Masked Patch Modeling
- **Baselines**: Classical forecasting methods (ARIMA, Prophet) used as comparison benchmarks

## Requirements

### Requirement 1: Project Structure and Configuration

**User Story:** As a beginner developer, I want a modular project structure with one file per component and all hyperparameters centralized, so that I can understand and modify each piece independently.

#### Acceptance Criteria

1. THE Project SHALL organize source code into separate directories: `data/`, `model/`, `pretraining/`, `forecasting/`, `evaluation/`, `utils/`, and `app/`
2. THE Project SHALL contain exactly one Python module per logical component (where a logical component is one of: data loading, patching, masking, model architecture, attention mechanism, pretraining loop, fine-tuning loop, forecasting inference, metric calculation, visualization, Gradio app entry point, and configuration) with no module exceeding 300 lines of code excluding comments and blank lines
3. THE Config_Module SHALL define all hyperparameters (D_MODEL=256, N_HEADS=8, N_LAYERS=6, PATCH_LEN=16, PATCH_STRIDE=8, DROPOUT=0.1, MASK_RATIO=0.4, PRETRAIN_LR=1e-4, PRETRAIN_EPOCHS=20, PRETRAIN_BATCH_SIZE=32, GRADIENT_ACCUMULATION=4, FORECAST_HORIZON=96, CONTEXT_LENGTH=512) in a single `config.py` file that is imported by all other modules requiring those values
4. THE Setup_Module SHALL install all dependencies with exact pinned version numbers (using `==` specifiers) via a `setup.py` file
5. THE Project SHALL include an inline comment above every function definition and every class definition, and at least one inline comment per 20 lines of non-comment code, explaining the purpose of the subsequent logic in plain language
6. IF any single module exceeds 300 lines of code excluding comments and blank lines, THEN THE Project SHALL split that module into sub-modules within the same directory such that each sub-module remains below 300 lines

### Requirement 2: Multi-Domain Data Acquisition

**User Story:** As a researcher, I want to download and verify real-world datasets from three domains, so that the model learns generalizable time series patterns.

#### Acceptance Criteria

1. WHEN the Energy download script is executed, THE Data_Pipeline SHALL download a multivariate electricity consumption dataset containing at least 2 numeric columns and at least 10,000 rows, and store it as a CSV file in the local `data/raw/` directory
2. WHEN the Weather download script is executed, THE Data_Pipeline SHALL download a multivariate weather observations dataset containing at least 2 numeric columns and at least 10,000 rows, and store it as a CSV file in the local `data/raw/` directory
3. WHEN the Finance download script is executed, THE Data_Pipeline SHALL download stock or crypto OHLCV data containing at least 5 columns (Open, High, Low, Close, Volume) and at least 5,000 rows, and store it as a CSV file in the local `data/raw/` directory
4. WHEN the ETTh1 download script is executed, THE Data_Pipeline SHALL download the ETTh1 benchmark dataset and store it as a CSV file in `data/raw/etth1/`
5. WHEN the verification script is executed, THE Data_Pipeline SHALL print dataset statistics including row count, column count, date range, missing value percentage, and file size in bytes for each dataset to standard output
6. IF a download fails due to network error, THEN THE Data_Pipeline SHALL retry up to 3 times with exponential backoff starting at 2 seconds (doubling each retry) and report an error message indicating the dataset name, URL, and failure reason on final failure
7. IF a dataset file already exists in the target directory, THEN THE Data_Pipeline SHALL skip the download and print a message indicating the file was already present
8. WHEN a download completes, THE Data_Pipeline SHALL verify that the downloaded file contains a parseable header row and at least 1,000 data rows, and IF verification fails, THEN THE Data_Pipeline SHALL delete the corrupt file and report an error message indicating the validation failure

### Requirement 3: Data Preprocessing and Patch Creation

**User Story:** As a model developer, I want raw time series normalized and split into patches, so that the transformer can process fixed-length input segments.

#### Acceptance Criteria

1. WHEN raw data is preprocessed, THE Data_Pipeline SHALL apply per-channel z-score normalization (subtract mean, divide by standard deviation) computed only on the training split
2. WHEN normalized data is patched, THE Data_Pipeline SHALL segment each time series into overlapping patches of length 16 with stride 8, producing floor((L - 16) / 8) + 1 patches for a series of length L
3. WHEN data is split, THE Data_Pipeline SHALL divide each dataset into train (70%), validation (15%), and test (15%) splits using chronological ordering without shuffling
4. THE Data_Pipeline SHALL store normalization statistics (mean and standard deviation per channel) as a JSON file in `data/processed/` for inverse transformation during evaluation
5. IF a time series is shorter than one Context_Window (512 time steps), THEN THE Data_Pipeline SHALL discard that series and log a warning indicating the series identifier and its length
6. IF a channel has zero standard deviation in the training split, THEN THE Data_Pipeline SHALL set its standard deviation to 1.0 to avoid division by zero and log a warning
7. WHEN trailing time steps remain after the last complete patch (fewer than PATCH_LEN steps), THE Data_Pipeline SHALL discard those trailing steps

### Requirement 4: PatchTST Model Architecture

**User Story:** As a researcher, I want a PatchTST transformer model that processes patched time series, so that I can pretrain it on multiple domains.

#### Acceptance Criteria

1. THE PatchTST_Model SHALL process each input channel independently (channel-independent design), accepting a univariate time series of Context_Window length (512 time steps) per forward pass and splitting it into patches of length 16 with stride 8 using a learnable linear projection to dimension 256, producing 63 patch embeddings
2. THE PatchTST_Model SHALL add learnable positional encodings of dimension 256 to each patch embedding before passing to the transformer encoder
3. THE PatchTST_Model SHALL process patch embeddings through 6 transformer encoder layers, each containing multi-head self-attention with 8 heads and a feedforward network with GELU activation and hidden dimension 4 times D_MODEL (1024)
4. THE PatchTST_Model SHALL apply dropout of 0.1 after attention and feedforward sublayers
5. THE PatchTST_Model SHALL apply layer normalization before each sublayer (pre-norm architecture)
6. WHEN the model receives an input of Context_Window length (512 time steps), THE PatchTST_Model SHALL produce 63 output embeddings of dimension 256, one for each patch position
7. THE PatchTST_Model SHALL have fewer than 10 million total trainable parameters to fit within Colab_Environment VRAM constraints
8. IF the input time series length is not compatible with the patch configuration (does not yield at least 1 patch of length 16), THEN THE PatchTST_Model SHALL raise an error indicating the minimum required input length

### Requirement 5: Masked Patch Modeling Pretraining

**User Story:** As a researcher, I want to pretrain the model by masking and reconstructing patches, so that it learns rich time series representations across domains.

#### Acceptance Criteria

1. WHEN a batch of patches is prepared for pretraining, THE Masking_Module SHALL randomly select 40% of patches per sample using uniform random sampling without replacement and replace them with a learnable mask token of dimension D_MODEL (256)
2. WHEN masked patches are processed, THE Pretraining_Loop SHALL compute Mean Squared Error loss averaged only over the masked patch positions (not on visible patches), comparing reconstructed patches against the normalized input patch values
3. THE Pretraining_Loop SHALL iterate over all three domain datasets (Energy, Weather, Finance) in each epoch using round-robin interleaved batching, drawing one batch from each dataset in rotation, where one epoch completes when the smallest dataset is fully consumed
4. THE Pretraining_Loop SHALL use the AdamW optimizer with learning rate 1e-4, weight decay 0.01, and gradient accumulation over 4 steps (effective batch size of 128)
5. THE Pretraining_Loop SHALL train for 20 epochs with a cosine learning rate schedule decaying to a minimum of 1e-6, with linear warmup over the first 2 epochs
6. WHEN any single training step takes longer than 20 minutes on Colab_Environment, THE Pretraining_Loop SHALL display a warning to standard output and save a Checkpoint automatically
7. THE Pretraining_Loop SHALL save a Checkpoint to Google Drive at the end of every epoch
8. THE Pretraining_Loop SHALL compute and log the average validation loss across all three domain validation splits at the end of every epoch
9. IF the training loss becomes NaN or exceeds 1e6 during any step, THEN THE Pretraining_Loop SHALL stop training, save a Checkpoint of the last valid state, and display an error message indicating training divergence

### Requirement 6: Probabilistic Forecasting Head

**User Story:** As a forecasting user, I want prediction intervals instead of point forecasts, so that I can quantify uncertainty in future values.

#### Acceptance Criteria

1. THE Probabilistic_Head SHALL accept patch embeddings from the PatchTST_Model encoder and output three quantile predictions (P10, P50, P90) for a single target channel at each time step in the Forecast_Horizon (96 steps), producing an output shape of (batch_size, 96, 3)
2. THE Probabilistic_Head SHALL use quantile regression loss (pinball loss) with quantile levels [0.1, 0.5, 0.9] during training, computed in the normalized scale and averaged across all three quantiles and all 96 time steps per sample
3. THE Probabilistic_Head SHALL ensure monotonicity: P10 prediction is less than or equal to P50 prediction, and P50 prediction is less than or equal to P90 prediction for every forecast step, with any violations corrected by sorting the three quantile values at each time step
4. WHEN the Probabilistic_Head produces forecasts for evaluation or inference, THE Probabilistic_Head SHALL apply inverse normalization using the per-channel mean and standard deviation stored by the Data_Pipeline to return predictions in the original data scale
5. IF the Probabilistic_Head receives input embeddings with a sequence length that does not correspond to a valid Context_Window of 512 time steps, THEN THE Probabilistic_Head SHALL raise an error indicating the expected input dimensions

### Requirement 7: Zero-Shot Forecasting Evaluation

**User Story:** As a researcher, I want to evaluate the pretrained model on ETTh1 without fine-tuning, so that I can demonstrate transfer learning capability.

#### Acceptance Criteria

1. WHEN zero-shot evaluation is triggered, THE Zero_Shot_Evaluator SHALL load the pretrained PatchTST_Model weights, set the model to inference mode (no dropout, no gradient computation), and perform no parameter updates on ETTh1_Dataset
2. WHEN zero-shot evaluation is triggered, THE Zero_Shot_Evaluator SHALL generate probabilistic forecasts (P10, P50, P90) for all test windows in ETTh1_Dataset using a sliding window with Context_Window of 512 steps, Forecast_Horizon of 96 steps, and a stride of 96 steps (non-overlapping forecast horizons)
3. THE Zero_Shot_Evaluator SHALL compute MAE and MSE using the P50 (median) quantile forecast, and CRPS using all three quantiles (P10, P50, P90), aggregated as the mean across all test windows and all forecast steps on the ETTh1_Dataset test split
4. THE Zero_Shot_Evaluator SHALL achieve lower mean MAE on ETTh1_Dataset (using P50 forecasts) than both ARIMA and Prophet Baselines computed on the same test windows
5. THE Zero_Shot_Evaluator SHALL report results in a comparison table with rows for PatchTST zero-shot, ARIMA, and Prophet, and columns for MAE, MSE, and CRPS metrics
6. IF the pretrained checkpoint file is not found or fails to load, THEN THE Zero_Shot_Evaluator SHALL raise an error message indicating the missing checkpoint path and abort evaluation without producing partial results

### Requirement 8: Classical Baselines

**User Story:** As a researcher, I want ARIMA and Prophet baselines on the same test set, so that I can fairly compare the foundation model's zero-shot performance.

#### Acceptance Criteria

1. WHEN baseline evaluation is triggered, THE Baselines module SHALL fit an ARIMA model using automatic order selection (auto ARIMA with maximum p=5, d=2, q=5) on the ETTh1_Dataset training split and generate point forecasts for each test window with Forecast_Horizon of 96 steps
2. WHEN baseline evaluation is triggered, THE Baselines module SHALL fit a Prophet model on the ETTh1_Dataset training split and generate point forecasts for each test window with Forecast_Horizon of 96 steps
3. THE Baselines module SHALL compute MAE, MSE, and MASE metrics for each baseline on the same test windows used by the Zero_Shot_Evaluator, where MASE is scaled by the mean absolute error of a seasonal naive forecast with a seasonal period of 24 time steps
4. IF ARIMA fitting fails to converge for a test window, THEN THE Baselines module SHALL use a seasonal naive forecast with a seasonal period of 24 time steps as fallback and log the window index and failure reason
5. IF Prophet fitting fails for a test window, THEN THE Baselines module SHALL use a seasonal naive forecast with a seasonal period of 24 time steps as fallback and log the window index and failure reason

### Requirement 9: Evaluation Metrics and Visualization

**User Story:** As a researcher, I want comprehensive metrics and visual comparisons, so that I can assess model quality and communicate results clearly.

#### Acceptance Criteria

1. THE Metrics_Module SHALL compute Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Scaled Error (MASE) using the P50 quantile prediction as the point forecast, and Continuous Ranked Probability Score (CRPS) using the full probabilistic forecast (P10, P50, P90), evaluated on the ETTh1_Dataset test split
2. THE Results_Table_Module SHALL print a formatted comparison table with rows for PatchTST zero-shot, PatchTST fine-tuned, ARIMA, and Prophet, and columns for MAE, MSE, MASE, and CRPS, with numeric values rounded to 4 decimal places
3. WHEN visualization is triggered, THE Visualization_Module SHALL generate plots showing actual values, P50 predictions, and shaded P10-P90 prediction intervals for at least 5 test windows selected by evenly spacing across the ETTh1_Dataset test split, where each plot includes axis labels, a descriptive title indicating the window index, and a legend distinguishing actual values from predictions
4. WHILE running in Colab_Environment, THE Visualization_Module SHALL display all plots inline; THE Visualization_Module SHALL save all plots as PNG files with a minimum resolution of 150 DPI to the `evaluation/` output directory

### Requirement 10: Colab Environment Utilities

**User Story:** As a Colab user, I want automatic checkpointing and resource monitoring, so that I do not lose progress when sessions reset.

#### Acceptance Criteria

1. WHEN `mount_drive()` is called, THE Colab_Helpers module SHALL mount Google Drive at `/content/drive/MyDrive/` and create a checkpoint directory named `checkpoints/` within the project root on Google Drive if it does not already exist
2. WHEN `save_checkpoint()` is called, THE Colab_Helpers module SHALL save model weights, optimizer state, current epoch, and training loss to a file on Google Drive named with a timestamp in `YYYYMMDD_HHMMSS` format, retaining a maximum of 5 most recent checkpoint files and deleting older ones
3. WHEN `load_checkpoint()` is called, THE Colab_Helpers module SHALL load the checkpoint file with the most recent filesystem modification time from Google Drive and restore model weights, optimizer state, and epoch counter
4. IF `load_checkpoint()` is called and no checkpoint file exists in the checkpoint directory, THEN THE Colab_Helpers module SHALL return None and print a message indicating that no checkpoint was found
5. WHEN `check_vram()` is called, THE Colab_Helpers module SHALL print current GPU memory usage (allocated and total) in megabytes and return a boolean indicating whether at least 2GB of free VRAM remains
6. IF `check_vram()` is called and no GPU is available, THEN THE Colab_Helpers module SHALL return False and print a message indicating that no GPU was detected
7. WHEN `session_timer()` is called, THE Colab_Helpers module SHALL return elapsed session time in minutes as a float and print a warning message if the session has been running for more than 10 hours
8. IF `save_checkpoint()` is called and the write operation fails, THEN THE Colab_Helpers module SHALL raise an IOError and print a message indicating the checkpoint save failed

### Requirement 11: Experiment Logging

**User Story:** As a researcher, I want training metrics logged to Weights & Biases, so that I can track experiments and compare runs.

#### Acceptance Criteria

1. WHEN pretraining begins, THE Logger module SHALL initialize a Weights & Biases run with project name, all hyperparameters from Config_Module, and a run name containing the dataset combination and a timestamp
2. WHEN an epoch ends, THE Logger module SHALL log training loss, validation loss, learning rate, and epoch number to the active Weights & Biases run
3. WHEN an epoch ends, THE Logger module SHALL log GPU memory allocated (in MB) and GPU memory reserved (in MB) to the active Weights & Biases run
4. IF Weights & Biases is not configured (no API key), THEN THE Logger module SHALL fall back to logging all metrics to a local CSV file in the project checkpoint directory and print a warning to stdout indicating that W&B is unavailable
5. IF a Weights & Biases logging call fails during training, THEN THE Logger module SHALL write the failed metrics to the local CSV fallback file and continue training without interruption

### Requirement 12: Fine-Tune Evaluation (Upper Bound)

**User Story:** As a researcher, I want to also fine-tune the pretrained model on ETTh1, so that I can show the upper bound of performance and demonstrate the value of pretraining.

#### Acceptance Criteria

1. WHEN fine-tune evaluation is triggered, THE Finetune_Evaluator SHALL load pretrained PatchTST_Model weights, freeze all transformer encoder parameters, and train only the Probabilistic_Head on ETTh1_Dataset training split for 10 epochs with a batch size of 32
2. THE Finetune_Evaluator SHALL use a learning rate of 1e-5 with a cosine schedule and linear warmup over the first epoch
3. WHEN fine-tuning completes, THE Finetune_Evaluator SHALL select the model checkpoint with the lowest validation loss on the ETTh1_Dataset validation split and compute MAE, MSE, and CRPS on the ETTh1_Dataset test split using the same test windows as the Zero_Shot_Evaluator
4. THE Finetune_Evaluator SHALL include fine-tuned results in the same comparison table as zero-shot and baseline results
5. IF training loss becomes NaN or validation loss increases for 3 consecutive epochs, THEN THE Finetune_Evaluator SHALL stop training early, restore the best checkpoint by validation loss, and log a warning

### Requirement 13: Interactive Demo Application

**User Story:** As a user, I want to upload a CSV and get probabilistic forecasts through a web interface, so that I can use the model without writing code.

#### Acceptance Criteria

1. THE Gradio_App SHALL provide a file upload widget accepting CSV files up to 50 MB in size, containing a datetime-parseable column and at least one numeric value column
2. WHEN a CSV is uploaded, THE Gradio_App SHALL populate a dropdown with all numeric columns from the uploaded file for target selection, and display a slider for selecting the Forecast_Horizon between 24 and 192 steps in increments of 1
3. WHEN the user clicks "Forecast", THE Gradio_App SHALL load the pretrained PatchTST_Model, run inference on the last Context_Window (512 time steps) of the selected target column, and display a plot containing the historical context series, the P50 forecast line, and a shaded region between P10 and P90, with a labeled x-axis (time steps) and y-axis (value), within 30 seconds
4. THE Gradio_App SHALL be deployable as a HuggingFace Space by running `app.py` as the entry point with all dependencies listed in `requirements.txt` and no additional manual configuration
5. IF the uploaded CSV has fewer rows than Context_Window (512), THEN THE Gradio_App SHALL display an error message stating the minimum required number of rows is 512
6. IF the uploaded CSV contains no datetime-parseable column or no numeric columns, THEN THE Gradio_App SHALL display an error message indicating which requirement is not met and not proceed to the column selection step
7. IF model inference fails or exceeds the 30-second time limit, THEN THE Gradio_App SHALL display an error message indicating the failure reason and allow the user to retry without re-uploading the file

### Requirement 14: Code Documentation and Readability

**User Story:** As a beginner coder, I want every piece of code explained with comments, so that I can learn while building.

#### Acceptance Criteria

1. THE Project SHALL include a module-level docstring in every Python file that states the file's purpose in one sentence and describes how it relates to other modules in the pipeline
2. THE Project SHALL include inline comments before every logical code block (minimum one comment per 10 lines of code)
3. THE Project SHALL include type hints on all function signatures including both parameter types and return types
4. THE README SHALL contain a step-by-step guide covering each pipeline stage (dependency installation, data download, data preprocessing, pretraining, zero-shot evaluation, baseline comparison, fine-tune evaluation, and demo deployment) with the exact command to run for each stage
5. THE Project SHALL include a docstring on every public function and class that describes what it does, its parameters, and its return value
