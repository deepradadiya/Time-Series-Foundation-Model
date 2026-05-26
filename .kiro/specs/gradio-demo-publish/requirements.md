# Requirements Document

## Introduction

This feature delivers the final three deliverables for the Time Series Foundation Model project: an enhanced multi-tab Gradio demo application with interactive Plotly charts and benchmark comparisons, a HuggingFace Hub publishing script that pushes models and deploys the Space, and a comprehensive final README for the GitHub repository. Together these make the project publicly accessible, reproducible, and portfolio-ready.

## Glossary

- **Gradio_App**: The interactive web application built with Gradio that serves the forecasting demo (located at `app/gradio_app.py`)
- **Publish_Script**: The Python script (`publish_to_hub.py`) that uploads models and deploys the Space to HuggingFace Hub
- **README_Generator**: The module or process that produces the final `README.md` for the GitHub repository
- **PatchTST_Model**: The pretrained PatchTST transformer encoder backbone used for time series forecasting
- **Forecast_Head**: The ProbabilisticForecastHead that produces P10/P50/P90 quantile forecasts
- **ETTh1**: The Electricity Transformer Temperature hourly dataset used as the benchmark
- **HuggingFace_Hub**: The HuggingFace model and Space hosting platform
- **Plotly_Chart**: An interactive chart rendered using the Plotly graphing library
- **Frequency_Detector**: The component that auto-detects time series frequency (hourly, daily, or weekly) from uploaded CSV data
- **Confidence_Band**: The shaded region between P10 and P90 quantile forecasts representing the 80% prediction interval
- **Model_Card**: A structured markdown document describing a model's purpose, training data, metrics, and usage

## Requirements

### Requirement 1: CSV Upload Tab with Auto-Detection and Interactive Forecast

**User Story:** As a user, I want to upload my own CSV data and receive an interactive probabilistic forecast, so that I can evaluate the model on my own time series.

#### Acceptance Criteria

1. WHEN a user uploads a CSV file containing at least one datetime-parseable column and at least one numeric column, THE Frequency_Detector SHALL auto-detect the frequency as one of hourly, daily, or weekly by computing the median timestamp interval across consecutive rows and classifying it to the nearest supported frequency
2. WHEN a CSV file is uploaded, THE Gradio_App SHALL allow the user to select a forecast horizon from the set of 24, 48, 96, or 192 steps via a radio button or dropdown
3. WHEN the user triggers a forecast, THE Gradio_App SHALL extract the last 512 time steps from the selected numeric column, normalize them using z-score normalization (mean and standard deviation computed from those 512 values), and run PatchTST_Model inference to produce quantile forecasts at P10, P50, and P90 levels
4. WHEN inference completes, THE Gradio_App SHALL display a Plotly_Chart with historical data rendered as a blue solid line, P50 forecast as an orange dashed line, and P10-P90 as a shaded light orange Confidence_Band labeled "80% confidence interval"
5. WHEN actual future values are available in the uploaded CSV for at least the full forecast horizon length beyond the 512-step context window, THE Gradio_App SHALL compute and display MAE and MASE metrics comparing the P50 forecast to actuals, where MASE uses a seasonal period of 24 time steps
6. IF the uploaded CSV contains fewer than 512 rows, THEN THE Gradio_App SHALL display an error message stating that the file must contain at least 512 data rows
7. IF the uploaded CSV contains no numeric columns, THEN THE Gradio_App SHALL display an error message indicating that at least one numeric column is required
8. IF the uploaded CSV contains no column parseable as datetime timestamps, THEN THE Gradio_App SHALL display an error message indicating that a datetime column is required for frequency detection
9. IF the median timestamp interval does not fall within a recognized tolerance of hourly, daily, or weekly frequency, THEN THE Frequency_Detector SHALL default to treating the data as daily frequency and display a warning indicating the assumed frequency

### Requirement 2: Live Benchmark Demo Tab

**User Story:** As a visitor, I want to see side-by-side comparisons of PatchTST against traditional baselines on real data, so that I can understand the model's relative performance.

#### Acceptance Criteria

1. WHEN the benchmark tab loads, THE Gradio_App SHALL present a dropdown containing 10 pre-loaded ETTh1 test samples identified by their starting index in the test split
2. WHEN a user selects a sample from the dropdown, THE Gradio_App SHALL display on a single Plotly_Chart the ground truth values for the 96-step forecast horizon, the pre-computed ARIMA point forecast, the pre-computed Prophet point forecast, and the PatchTST_Model zero-shot forecast with a shaded P10–P90 confidence band
3. WHEN the comparison chart is rendered, THE Gradio_App SHALL label each forecast method in the chart legend with its name and display the per-sample MAE value for each method, visually distinguishing the method that achieves the lowest MAE by appending a best-indicator label to its legend entry
4. THE Gradio_App SHALL pre-compute and cache ARIMA and Prophet forecasts for the 10 benchmark samples at application startup or build time so that selecting a sample renders the chart within 2 seconds

### Requirement 3: About the Model Tab

**User Story:** As a visitor, I want to understand the model architecture and capabilities at a glance, so that I can assess its suitability for my use case.

#### Acceptance Criteria

1. THE Gradio_App SHALL display an ASCII architecture diagram of the PatchTST_Model in a fixed-width text box showing the data flow from multi-domain input through patching (16-step windows), masked patch modeling, transformer encoder (6 layers, 256 dim, 8 heads), to P10/P50/P90 quantile output
2. THE Gradio_App SHALL list the three pretraining domains (Energy, Weather, Finance) with a one-sentence description of each domain's data source
3. THE Gradio_App SHALL display a metrics table showing key benchmark results (MAE, MSE, MASE, CRPS) for PatchTST zero-shot, PatchTST fine-tuned, ARIMA, and Prophet on the ETTh1 dataset
4. THE Gradio_App SHALL provide clickable links to the HuggingFace_Hub model page and the GitHub repository, opening in a new browser tab

### Requirement 4: HuggingFace Hub Publishing Script

**User Story:** As the developer, I want a single script that publishes all artifacts to HuggingFace Hub, so that the model and demo are publicly accessible with minimal manual steps.

#### Acceptance Criteria

1. WHEN executed, THE Publish_Script SHALL push the pretrained PatchTST_Model checkpoint from `checkpoints/pretrained_patchtst.pt` to a HuggingFace_Hub repository named "{username}/patchtst-foundation-pretrained", creating the repository if it does not already exist
2. WHEN executed, THE Publish_Script SHALL push the fine-tuned ETTh1 model checkpoint from `checkpoints/finetuned_patchtst.pt` to a HuggingFace_Hub repository named "{username}/patchtst-etth1-finetuned", creating the repository if it does not already exist
3. WHEN executed, THE Publish_Script SHALL deploy the Gradio_App as a HuggingFace Space named "{username}/timeseries-foundation-demo" by uploading the application source files and `requirements.txt`, with the Space SDK configured as Gradio
4. WHEN publishing a model, THE Publish_Script SHALL generate and upload a Model_Card containing: model name and architecture summary with parameter count, pretraining domains with row counts per domain, a zero-shot benchmark results table with MAE/MSE/MASE/CRPS values, a 5-line Python usage example, and a link to the GitHub repository
5. IF the HF_TOKEN environment variable is not set or is empty, THEN THE Publish_Script SHALL print an error message to stderr instructing the user to set the HF_TOKEN environment variable and exit with a non-zero status code without attempting any uploads
6. IF a network error occurs during upload, THEN THE Publish_Script SHALL retry up to 3 times with exponential backoff starting at 2 seconds (doubling each retry) before printing the failure reason to stderr and exiting with a non-zero status code
7. IF a required checkpoint file does not exist at the expected path, THEN THE Publish_Script SHALL print an error message to stderr identifying the missing file and exit with a non-zero status code without attempting any uploads
8. WHEN executed, THE Publish_Script SHALL resolve "{username}" by querying the HuggingFace Hub API using the configured HF_TOKEN to obtain the authenticated user's username

### Requirement 5: Final README Generation

**User Story:** As a portfolio reviewer or recruiter, I want a clear README that explains what this project is, shows results, and tells me how to reproduce it, so that I can evaluate the developer's skills.

#### Acceptance Criteria

1. THE README_Generator SHALL produce a "What This Is" section containing exactly 3 sentences, where each sentence uses no domain-specific acronyms without inline definitions and contains no more than 25 words
2. THE README_Generator SHALL include an ASCII architecture diagram section showing the model's data flow from input shape through patching, projection, transformer encoder, to probabilistic output shape, matching the architecture description present in the Gradio_App About tab
3. THE README_Generator SHALL include a Results section with a formatted markdown table populated from `final_metrics.json` showing columns MAE, MSE, MASE, and CRPS with values rounded to 4 decimal places, and rows in the order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned), displaying "N/A" for CRPS on point-forecast-only baselines (Naive, ARIMA, Prophet)
4. THE README_Generator SHALL include a Pretraining Details section listing the three pretraining domain dataset names (Energy, Weather, Finance), the total training steps computed as epochs multiplied by steps-per-epoch, and an embedded markdown image reference to the pretraining loss curves PNG file
5. THE README_Generator SHALL include a How to Reproduce section containing one numbered command per pipeline stage covering all 8 stages: dependency installation, data download, data preprocessing, pretraining, zero-shot evaluation, baseline comparison, fine-tune evaluation, and demo deployment, where each command is a valid shell or Python command executable in Google Colab
6. THE README_Generator SHALL include a Resume Bullet section containing a single bullet point that includes the PatchTST zero-shot MAE value and at least one baseline MAE value from `final_metrics.json`, expressed as a comparative statement
7. THE README_Generator SHALL include a Links section with exactly 4 placeholder or actual URLs labeled as: HuggingFace pretrained model, HuggingFace fine-tuned model, HuggingFace Space demo, and W&B training run

### Requirement 6: Gradio App Structure and Navigation

**User Story:** As a user, I want a well-organized multi-tab interface, so that I can easily navigate between uploading data, viewing benchmarks, and learning about the model.

#### Acceptance Criteria

1. THE Gradio_App SHALL organize its interface into exactly three tabs labeled "Upload your own data", "Live benchmark demo", and "About the model", with "Upload your own data" selected as the active tab on initial load
2. THE Gradio_App SHALL use Plotly for all chart rendering within the "Upload your own data" and "Live benchmark demo" tabs, providing interactive zoom, pan, and hover tooltips on each rendered chart
3. WHEN the application starts, THE Gradio_App SHALL load the PatchTST_Model checkpoint once and cache it in memory for all subsequent inference requests, completing model loading within 60 seconds
4. IF no model checkpoint is found at startup, THEN THE Gradio_App SHALL initialize with random weights and display a persistent warning banner at the top of the "Upload your own data" tab indicating that predictions use an untrained model
5. IF a model checkpoint file exists but fails to load due to corruption or incompatible format, THEN THE Gradio_App SHALL fall back to random weight initialization and display a persistent warning banner at the top of the "Upload your own data" tab indicating that the checkpoint could not be loaded and predictions use an untrained model
6. IF model loading exceeds 60 seconds, THEN THE Gradio_App SHALL abort the loading attempt, initialize with random weights, and display a persistent warning banner indicating a loading timeout occurred
