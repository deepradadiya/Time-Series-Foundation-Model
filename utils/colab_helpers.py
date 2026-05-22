"""Colab environment utilities for Google Drive checkpointing and resource monitoring.

This module provides helper functions for running the Time Series Foundation Model
on Google Colab free tier. It handles mounting Google Drive for persistent storage,
saving/loading model checkpoints with automatic rotation, monitoring GPU VRAM usage,
and tracking session time to avoid unexpected resets.

Related modules:
    - pretraining/train.py uses save_checkpoint and load_checkpoint for training persistence
    - pretraining/train.py uses check_vram to monitor GPU memory during training
    - config.py provides hyperparameters referenced during checkpoint saving
"""

import os
import glob
import time
from datetime import datetime
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Module-level state: session start time is recorded when this module loads.
# This allows session_timer() to compute elapsed time from the moment the
# Colab notebook first imports this module.
# ---------------------------------------------------------------------------
_SESSION_START_TIME: float = time.time()

# Default checkpoint directory path on Google Drive
_CHECKPOINT_DIR: str = "/content/drive/MyDrive/checkpoints/"


def mount_drive() -> None:
    """Mount Google Drive and create the checkpoint directory.

    This function mounts Google Drive at /content/drive/MyDrive/ using the
    google.colab library (only available in Colab environments). It then
    creates a 'checkpoints/' directory on Drive if it does not already exist.

    Parameters:
        None

    Returns:
        None

    Raises:
        ImportError: If google.colab is not available (not running in Colab).
    """
    # Attempt to import the Colab-specific drive module
    from google.colab import drive

    # Mount Google Drive at the standard Colab mount point
    drive.mount("/content/drive")

    # Create the checkpoint directory if it doesn't exist yet
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    # Confirm successful mount and directory creation
    print(f"Google Drive mounted. Checkpoint directory: {_CHECKPOINT_DIR}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    max_keep: int = 5,
) -> str:
    """Save a training checkpoint to Google Drive with timestamp naming.

    Saves model weights, optimizer state, epoch number, and training loss
    to a .pt file named with the current timestamp (YYYYMMDD_HHMMSS format).
    Automatically deletes older checkpoints to retain only the most recent
    `max_keep` files.

    Parameters:
        model: The PyTorch model whose state_dict will be saved.
        optimizer: The optimizer whose state_dict will be saved.
        epoch: The current training epoch number.
        loss: The current training loss value.
        max_keep: Maximum number of checkpoint files to retain (default 5).

    Returns:
        The file path of the saved checkpoint.

    Raises:
        IOError: If the checkpoint file cannot be written (e.g., Drive full).
    """
    # Generate a timestamp string for the checkpoint filename
    # Format: YYYYMMDD_HHMMSS as specified in requirements (Requirement 10.2)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the full checkpoint file path
    checkpoint_path: str = os.path.join(_CHECKPOINT_DIR, f"checkpoint_{timestamp}.pt")

    # Build the checkpoint dictionary with all training state
    checkpoint_data: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": loss,
        "timestamp": timestamp,
    }

    # Attempt to save the checkpoint; raise IOError on failure
    try:
        # Ensure the checkpoint directory exists (in case it was deleted)
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

        # Save the checkpoint dictionary to disk
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} (epoch {epoch}, loss {loss:.6f})")
    except (OSError, RuntimeError) as e:
        # Print failure message and raise IOError as specified by requirements
        print(f"Checkpoint save failed: {checkpoint_path} — {e}")
        raise IOError(f"Failed to save checkpoint to {checkpoint_path}: {e}") from e

    # Enforce the max_keep limit by deleting the oldest checkpoints
    _cleanup_old_checkpoints(max_keep)

    return checkpoint_path


def _cleanup_old_checkpoints(max_keep: int) -> None:
    """Remove old checkpoint files, keeping only the most recent max_keep.

    Parameters:
        max_keep: Number of most recent checkpoint files to retain.

    Returns:
        None
    """
    # Find all checkpoint files matching our naming pattern
    pattern: str = os.path.join(_CHECKPOINT_DIR, "checkpoint_*.pt")
    checkpoint_files: list = sorted(glob.glob(pattern))

    # If we have more than max_keep files, delete the oldest ones
    if len(checkpoint_files) > max_keep:
        # Files are sorted alphabetically by timestamp, so oldest come first
        files_to_delete: list = checkpoint_files[: len(checkpoint_files) - max_keep]
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                print(f"Deleted old checkpoint: {filepath}")
            except OSError as e:
                # Non-critical: warn but don't fail if deletion fails
                print(f"Warning: Could not delete old checkpoint {filepath}: {e}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Optional[dict]:
    """Load the most recent checkpoint from Google Drive.

    Finds the checkpoint file with the most recent filesystem modification
    time in the checkpoint directory, then restores model weights, optimizer
    state, and returns the checkpoint metadata (epoch, loss, timestamp).

    Parameters:
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into.

    Returns:
        A dictionary with keys 'epoch', 'train_loss', and 'timestamp' if a
        checkpoint was found and loaded successfully. Returns None if no
        checkpoint files exist in the checkpoint directory.
    """
    # Find all checkpoint files in the directory
    pattern: str = os.path.join(_CHECKPOINT_DIR, "checkpoint_*.pt")
    checkpoint_files: list = glob.glob(pattern)

    # If no checkpoints exist, return None as specified
    if not checkpoint_files:
        print("No checkpoint found. Starting from scratch.")
        return None

    # Sort by modification time to find the most recent checkpoint
    most_recent: str = max(checkpoint_files, key=os.path.getmtime)

    # Load the checkpoint data from disk
    checkpoint_data: dict = torch.load(most_recent, map_location="cpu")

    # Restore model weights from the checkpoint
    model.load_state_dict(checkpoint_data["model_state_dict"])

    # Restore optimizer state from the checkpoint
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    print(
        f"Checkpoint loaded: {most_recent} "
        f"(epoch {checkpoint_data['epoch']}, loss {checkpoint_data['train_loss']:.6f})"
    )

    # Return metadata for the caller to use (e.g., resume epoch counter)
    return {
        "epoch": checkpoint_data["epoch"],
        "train_loss": checkpoint_data["train_loss"],
        "timestamp": checkpoint_data["timestamp"],
    }


def check_vram() -> bool:
    """Check GPU VRAM usage and report whether sufficient memory is available.

    Prints current GPU memory allocation (allocated and total) in megabytes.
    Returns True if at least 2 GB of free VRAM remains, False otherwise.
    If no GPU is available, prints a warning and returns False.

    Parameters:
        None

    Returns:
        True if a GPU is available and has >= 2 GB free VRAM, False otherwise.
    """
    # Check if CUDA (GPU) is available on this machine
    if not torch.cuda.is_available():
        print("No GPU detected. VRAM check unavailable.")
        return False

    # Query current GPU memory statistics (in bytes)
    allocated_bytes: int = torch.cuda.memory_allocated()
    total_bytes: int = torch.cuda.get_device_properties(0).total_mem

    # Convert bytes to megabytes for human-readable output
    allocated_mb: float = allocated_bytes / (1024 ** 2)
    total_mb: float = total_bytes / (1024 ** 2)
    free_mb: float = total_mb - allocated_mb

    # Print current memory usage summary
    print(
        f"GPU VRAM: {allocated_mb:.1f} MB allocated / {total_mb:.1f} MB total "
        f"({free_mb:.1f} MB free)"
    )

    # Check if at least 2 GB (2048 MB) of free VRAM remains
    has_sufficient_vram: bool = free_mb >= 2048.0

    # Warn if VRAM is running low
    if not has_sufficient_vram:
        print("Warning: Less than 2 GB of free VRAM remaining!")

    return has_sufficient_vram


def session_timer() -> float:
    """Return elapsed session time in minutes and warn if session is long.

    Computes the time elapsed since this module was first imported (which
    approximates the Colab session start). Prints a warning if the session
    has been running for more than 10 hours, since Colab free tier sessions
    reset after approximately 12 hours.

    Parameters:
        None

    Returns:
        Elapsed session time in minutes as a float.
    """
    # Calculate elapsed time since module import (session start proxy)
    elapsed_seconds: float = time.time() - _SESSION_START_TIME

    # Convert to minutes for the return value
    elapsed_minutes: float = elapsed_seconds / 60.0

    # Warn if session has been running for more than 10 hours (600 minutes)
    if elapsed_minutes > 600.0:
        print(
            f"Warning: Session has been running for {elapsed_minutes:.1f} minutes "
            f"({elapsed_minutes / 60.0:.1f} hours). "
            "Colab sessions reset after ~12 hours. Consider saving your work."
        )

    return elapsed_minutes
