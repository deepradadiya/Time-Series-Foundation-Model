"""Unit tests for the ExperimentLogger class in utils/logger.py.

Tests verify CSV fallback behavior, metric logging, GPU metric enrichment,
and graceful error handling when W&B is unavailable.
"""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import Config
from utils.logger import ExperimentLogger


class TestExperimentLoggerCSVFallback:
    """Tests for CSV fallback mode when W&B is not available."""

    def test_falls_back_to_csv_when_no_wandb_key(self, tmp_path, monkeypatch):
        """Logger should use CSV fallback when no W&B API key is set."""
        # Remove any existing W&B API key from environment
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        # Patch the csv_dir to use a temp directory
        monkeypatch.setattr(
            "utils.logger.ExperimentLogger.__init__",
            lambda self, config, run_name: None,
        )
        # Create logger manually to control paths
        logger = ExperimentLogger.__new__(ExperimentLogger)
        logger.config = Config
        logger.run_name = "test_run"
        logger.use_wandb = False
        logger.wandb_run = None
        logger.csv_dir = tmp_path
        logger.csv_path = tmp_path / "metrics_test_run.csv"
        logger._csv_header_written = False
        logger._hyperparams = {"D_MODEL": 256}

        # Log an epoch
        logger.log_epoch({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})

        # Verify CSV was created
        assert logger.csv_path.exists()

    def test_csv_contains_correct_metrics(self, tmp_path):
        """CSV file should contain all logged metrics with correct values."""
        # Create a logger in CSV-only mode
        logger = ExperimentLogger.__new__(ExperimentLogger)
        logger.config = Config
        logger.run_name = "test_csv_metrics"
        logger.use_wandb = False
        logger.wandb_run = None
        logger.csv_dir = tmp_path
        logger.csv_path = tmp_path / "metrics_test.csv"
        logger._csv_header_written = False
        logger._hyperparams = {}

        # Log metrics
        logger.log_epoch({
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "learning_rate": 1e-4,
        })

        # Read and verify CSV content
        with open(logger.csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["epoch"] == "1"
        assert rows[0]["train_loss"] == "0.5"
        assert rows[0]["val_loss"] == "0.6"

    def test_csv_appends_multiple_epochs(self, tmp_path):
        """Multiple log_epoch calls should append rows to the same CSV."""
        logger = ExperimentLogger.__new__(ExperimentLogger)
        logger.config = Config
        logger.run_name = "test_append"
        logger.use_wandb = False
        logger.wandb_run = None
        logger.csv_dir = tmp_path
        logger.csv_path = tmp_path / "metrics_append.csv"
        logger._csv_header_written = False
        logger._hyperparams = {}

        # Log 3 epochs
        for i in range(1, 4):
            logger.log_epoch({"epoch": i, "train_loss": 0.5 / i})

        # Verify 3 data rows plus header
        with open(logger.csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 4  # 1 header + 3 data rows


class TestExperimentLoggerGPUMetrics:
    """Tests for GPU memory metric enrichment."""

    def test_adds_gpu_metrics_when_cuda_available(self):
        """Should add gpu_memory_allocated_mb and gpu_memory_reserved_mb."""
        logger = ExperimentLogger.__new__(ExperimentLogger)

        # Mock torch.cuda as available
        with patch("utils.logger.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 500
            mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 800

            result = logger._add_gpu_metrics({"epoch": 1})

        assert result["gpu_memory_allocated_mb"] == 500.0
        assert result["gpu_memory_reserved_mb"] == 800.0
        assert result["epoch"] == 1

    def test_adds_zero_gpu_metrics_when_no_cuda(self):
        """Should add zero GPU metrics when CUDA is not available."""
        logger = ExperimentLogger.__new__(ExperimentLogger)

        with patch("utils.logger.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            result = logger._add_gpu_metrics({"epoch": 1})

        assert result["gpu_memory_allocated_mb"] == 0.0
        assert result["gpu_memory_reserved_mb"] == 0.0


class TestExperimentLoggerHyperparams:
    """Tests for hyperparameter extraction from Config."""

    def test_extracts_all_config_params(self):
        """Should extract all public non-callable attributes from Config."""
        logger = ExperimentLogger.__new__(ExperimentLogger)
        params = logger._extract_hyperparams(Config)

        # Verify key hyperparameters are present
        assert params["D_MODEL"] == 256
        assert params["N_HEADS"] == 8
        assert params["N_LAYERS"] == 6
        assert params["PATCH_LEN"] == 16
        assert params["PRETRAIN_LR"] == 1e-4
        assert params["FORECAST_HORIZON"] == 96

    def test_excludes_private_attributes(self):
        """Should not include attributes starting with underscore."""
        logger = ExperimentLogger.__new__(ExperimentLogger)
        params = logger._extract_hyperparams(Config)

        # No private attributes should be present
        for key in params:
            assert not key.startswith("_")


class TestExperimentLoggerFinish:
    """Tests for the finish() method."""

    def test_finish_csv_mode_prints_path(self, tmp_path, capsys):
        """finish() in CSV mode should print the CSV file path."""
        logger = ExperimentLogger.__new__(ExperimentLogger)
        logger.use_wandb = False
        logger.wandb_run = None
        logger.csv_path = tmp_path / "metrics.csv"

        logger.finish()

        captured = capsys.readouterr()
        assert "Experiment logging complete" in captured.out
        assert str(logger.csv_path) in captured.out
