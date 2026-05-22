"""Experiment logging module with Weights & Biases integration and CSV fallback.

This module provides the ExperimentLogger class that logs training metrics
(loss, learning rate, GPU memory, etc.) to Weights & Biases for experiment
tracking. If W&B is unavailable (no API key configured), it gracefully falls
back to writing metrics to a local CSV file so training is never interrupted.

Related modules:
    - config.py provides hyperparameters logged at run initialization
    - pretraining/train.py calls log_epoch() after each training epoch
    - forecasting/finetune.py uses the same logger for fine-tuning metrics
    - utils/colab_helpers.py provides GPU memory info logged alongside metrics
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Try importing torch for GPU memory reporting
try:
    import torch
except ImportError:
    torch = None  # type: ignore

# Try importing wandb — it may not be installed or configured
try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

# Import project configuration for hyperparameters
from config import Config


class ExperimentLogger:
    """Weights & Biases logger with automatic CSV fallback.

    This class initializes a W&B run when an API key is available, logging
    all training metrics (loss, learning rate, GPU memory) to the cloud
    dashboard. If W&B is not configured or a logging call fails, metrics
    are written to a local CSV file instead, ensuring training continues
    without interruption.

    Attributes:
        config: The Config class containing all hyperparameters.
        run_name: A descriptive name for this experiment run.
        use_wandb: Whether W&B logging is active (True) or using CSV fallback (False).
        csv_path: Path to the local CSV fallback file.
        wandb_run: The active W&B run object (None if using CSV fallback).

    Parameters:
        config: Config class with all hyperparameters to log.
        run_name: Name for the experiment run (e.g., "energy_weather_finance_20240101_120000").

    Returns:
        None
    """

    def __init__(self, config: type, run_name: str) -> None:
        """Initialize the experiment logger with W&B or CSV fallback.

        Attempts to initialize a Weights & Biases run. If W&B is not installed
        or no API key is found, falls back to local CSV logging and prints a
        warning to stdout.

        Args:
            config: The Config class containing all hyperparameters.
            run_name: A descriptive name for this experiment run.
        """
        # Store configuration and run metadata
        self.config = config
        self.run_name = run_name
        self.use_wandb: bool = False
        self.wandb_run: Optional[Any] = None

        # Set up the CSV fallback path in the checkpoints directory
        # This ensures metrics are always persisted locally regardless of W&B status
        self.csv_dir: Path = Path("checkpoints")
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path: Path = self.csv_dir / f"metrics_{run_name}.csv"

        # Track whether the CSV header has been written
        self._csv_header_written: bool = False

        # Collect all hyperparameters from Config as a dictionary for logging
        self._hyperparams: Dict[str, Any] = self._extract_hyperparams(config)

        # Attempt to initialize Weights & Biases
        self._init_wandb()

    def _extract_hyperparams(self, config: type) -> Dict[str, Any]:
        """Extract all hyperparameters from the Config class as a dictionary.

        Iterates over Config class attributes (excluding private/dunder attributes)
        and collects them into a flat dictionary suitable for W&B config logging.

        Args:
            config: The Config class with hyperparameter class attributes.

        Returns:
            Dictionary mapping parameter names to their values.
        """
        # Collect all public class attributes that are not methods
        params: Dict[str, Any] = {}
        for attr_name in dir(config):
            # Skip private attributes and built-in methods
            if attr_name.startswith("_"):
                continue
            value = getattr(config, attr_name)
            # Only include data attributes, not methods
            if not callable(value):
                params[attr_name] = value
        return params

    def _init_wandb(self) -> None:
        """Attempt to initialize a Weights & Biases run.

        Checks if wandb is installed and an API key is available. If both
        conditions are met, initializes a new W&B run with the project name,
        hyperparameters, and run name. Otherwise, prints a warning and
        activates CSV fallback mode.
        """
        # Check if wandb module is available
        if wandb is None:
            print(
                "[WARNING] wandb is not installed. "
                "Falling back to local CSV logging."
            )
            self.use_wandb = False
            return

        # Check if a W&B API key is configured in the environment
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            # Also check if wandb is logged in via the CLI
            try:
                # wandb.api.api_key returns the key if logged in
                if not wandb.api.api_key:
                    raise ValueError("No API key")
            except (AttributeError, ValueError):
                print(
                    "[WARNING] Weights & Biases API key not configured. "
                    "Falling back to local CSV logging at: "
                    f"{self.csv_path}"
                )
                self.use_wandb = False
                return

        # Initialize the W&B run with project name and hyperparameters
        try:
            self.wandb_run = wandb.init(
                project="time-series-foundation-model",
                name=self.run_name,
                config=self._hyperparams,
                reinit=True,
            )
            self.use_wandb = True
            print(
                f"[INFO] Weights & Biases run initialized: {self.run_name}"
            )
        except Exception as e:
            # If W&B initialization fails for any reason, fall back to CSV
            print(
                f"[WARNING] Failed to initialize W&B: {e}. "
                f"Falling back to local CSV logging at: {self.csv_path}"
            )
            self.use_wandb = False

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for a completed epoch.

        Logs training loss, validation loss, learning rate, epoch number,
        and GPU memory usage. If W&B is active, logs to the cloud dashboard.
        If W&B fails or is unavailable, writes to the local CSV file.

        The metrics dictionary should contain keys like:
            - "epoch": Current epoch number (int)
            - "train_loss": Training loss for this epoch (float)
            - "val_loss": Validation loss for this epoch (float)
            - "learning_rate": Current learning rate (float)
            - Any additional metrics to track

        GPU memory (allocated and reserved in MB) is automatically appended
        if a CUDA GPU is available.

        Args:
            metrics: Dictionary of metric names to values for this epoch.
        """
        # Automatically add GPU memory metrics if a GPU is available
        metrics_with_gpu = self._add_gpu_metrics(metrics)

        # Add a timestamp for when this epoch was logged
        metrics_with_gpu["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Attempt W&B logging if active
        if self.use_wandb:
            try:
                wandb.log(metrics_with_gpu)
            except Exception as e:
                # W&B logging failed — write to CSV fallback and continue
                print(
                    f"[WARNING] W&B logging failed: {e}. "
                    "Writing metrics to CSV fallback."
                )
                self._write_csv(metrics_with_gpu)
        else:
            # W&B not available — always write to CSV
            self._write_csv(metrics_with_gpu)

    def _add_gpu_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Append GPU memory usage metrics to the metrics dictionary.

        Queries CUDA for current allocated and reserved memory in megabytes.
        If no GPU is available, GPU metrics are set to 0.

        Args:
            metrics: Existing metrics dictionary to augment.

        Returns:
            New dictionary with GPU memory metrics added.
        """
        # Create a copy to avoid mutating the caller's dictionary
        enriched = dict(metrics)

        # Check if CUDA is available for GPU memory reporting
        if torch is not None and torch.cuda.is_available():
            # Get allocated memory (actively used by tensors) in MB
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            # Get reserved memory (cached by allocator) in MB
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            enriched["gpu_memory_allocated_mb"] = round(allocated_mb, 2)
            enriched["gpu_memory_reserved_mb"] = round(reserved_mb, 2)
        else:
            # No GPU available — report zeros
            enriched["gpu_memory_allocated_mb"] = 0.0
            enriched["gpu_memory_reserved_mb"] = 0.0

        return enriched

    def _write_csv(self, metrics: Dict[str, Any]) -> None:
        """Write a single row of metrics to the local CSV fallback file.

        Creates the CSV file with a header row on the first write. Subsequent
        calls append rows. This ensures metrics are never lost even if W&B
        is unavailable or fails mid-training.

        Args:
            metrics: Dictionary of metric names to values for one epoch.
        """
        # Determine if we need to write the header (first write or new file)
        file_exists = self.csv_path.exists()
        write_header = not file_exists or not self._csv_header_written

        try:
            # Open in append mode to add rows without overwriting
            with open(self.csv_path, mode="a", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=sorted(metrics.keys())
                )
                # Write header only on the first call
                if write_header:
                    writer.writeheader()
                    self._csv_header_written = True
                # Write the metrics as a single row
                writer.writerow(metrics)
        except OSError as e:
            # If even CSV writing fails, print error but don't crash training
            print(
                f"[ERROR] Failed to write metrics to CSV at {self.csv_path}: {e}"
            )

    def finish(self) -> None:
        """Finalize the logging session.

        Closes the W&B run if active, ensuring all buffered metrics are
        flushed to the server. For CSV logging, no explicit close is needed
        since we open/close the file on each write.
        """
        # Close the W&B run to flush remaining data
        if self.use_wandb and self.wandb_run is not None:
            try:
                wandb.finish()
                print("[INFO] W&B run finished successfully.")
            except Exception as e:
                print(f"[WARNING] Error finishing W&B run: {e}")
        else:
            # CSV mode — just confirm logging location
            print(
                f"[INFO] Experiment logging complete. "
                f"Metrics saved to: {self.csv_path}"
            )
