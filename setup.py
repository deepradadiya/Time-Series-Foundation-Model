"""Setup module for the Time Series Foundation Model project.

This file defines all project dependencies with exact pinned versions to ensure
reproducible installations across environments. The project targets Google Colab
free tier (T4 GPU, ~15GB VRAM) and includes libraries for deep learning (PyTorch),
data processing (numpy, pandas), visualization (matplotlib), experiment tracking
(wandb), classical baselines (statsmodels, prophet), property-based testing
(hypothesis), and interactive deployment (gradio).

Related modules:
    - config.py: Central hyperparameter configuration imported by all modules
    - requirements.txt: Minimal runtime dependencies for HuggingFace Space deployment
"""

from setuptools import setup, find_packages

# Project metadata and dependency specification
setup(
    name="time-series-foundation-model",
    version="0.1.0",
    description="PatchTST-based Time Series Foundation Model with zero-shot forecasting",
    python_requires=">=3.9",
    packages=find_packages(),

    # All dependencies pinned with exact versions (==) for reproducibility
    install_requires=[
        # Deep learning framework — provides tensor operations, autograd, and GPU acceleration
        "torch==2.6.0",

        # Numerical computing — array operations for data preprocessing and metrics
        "numpy==2.2.0",

        # Data manipulation — CSV loading, time series handling, and DataFrame operations
        "pandas==2.2.3",

        # Visualization — plotting forecasts, prediction intervals, and training curves
        "matplotlib==3.10.0",

        # Machine learning utilities — preprocessing, metrics, and model selection helpers
        "scikit-learn==1.6.1",

        # Property-based testing — generates random inputs to validate correctness properties
        "hypothesis==6.122.3",

        # Test framework — runs unit tests, property tests, and integration tests
        "pytest==8.3.4",

        # Web UI framework — builds the interactive forecasting demo for HuggingFace Spaces
        "gradio==5.12.0",

        # Experiment tracking — logs training metrics, hyperparameters, and GPU usage
        "wandb==0.19.1",

        # Statistical models — provides ARIMA and other classical time series methods
        "statsmodels==0.14.4",

        # Facebook Prophet — classical baseline for time series forecasting comparison
        "prophet==1.1.6",

        # HuggingFace Hub — model hosting and Space deployment utilities
        "huggingface_hub==0.27.1",

        # Progress bars — displays training progress and data download status
        "tqdm==4.67.1",
    ],
)
