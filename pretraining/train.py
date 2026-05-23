"""Full pretraining loop for Masked Patch Modeling across multiple domains.

This module implements the complete pretraining pipeline for the PatchTST model.
It trains the model using Masked Patch Modeling (MPM) on three domains (Energy,
Weather, Finance) with round-robin interleaved batching. The training loop uses
AdamW optimizer with cosine learning rate scheduling, gradient accumulation,
and automatic checkpointing to Google Drive.

Related modules:
    - model/patchtst.py provides the PatchTSTModel (encoder backbone)
    - pretraining/masking.py provides PatchMasker for random patch masking
    - pretraining/reconstruction_head.py provides ReconstructionHead and loss function
    - data/dataset.py provides MultiDomainDataLoader for round-robin batching
    - config.py supplies all training hyperparameters
    - utils/colab_helpers.py provides checkpoint save/load and VRAM monitoring
    - utils/logger.py provides ExperimentLogger for W&B + CSV logging
"""

import math
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from model.patchtst import PatchTSTModel
from pretraining.masking import PatchMasker
from pretraining.reconstruction_head import (
    ReconstructionHead,
    compute_masked_reconstruction_loss,
)
from data.dataset import MultiDomainDataLoader, TimeSeriesDataset

# ---------------------------------------------------------------------------
# Conditional imports for utility modules that may not exist yet.
# These modules are implemented in later tasks, so we gracefully handle
# their absence by providing fallback behavior (print-based logging,
# no checkpointing to Drive).
# ---------------------------------------------------------------------------
try:
    from utils.colab_helpers import save_checkpoint, load_checkpoint, check_vram
    _HAS_COLAB_HELPERS = True
except ImportError:
    _HAS_COLAB_HELPERS = False

try:
    from utils.logger import ExperimentLogger
    _HAS_LOGGER = True
except ImportError:
    _HAS_LOGGER = False


# ---------------------------------------------------------------------------
# Step timeout threshold in seconds (20 minutes = 1200 seconds).
# If a single training step exceeds this duration, a warning is printed
# and an automatic checkpoint is saved to prevent data loss.
# ---------------------------------------------------------------------------
_STEP_TIMEOUT_SECONDS: int = 20 * 60

# ---------------------------------------------------------------------------
# Divergence threshold: if training loss exceeds this value, training stops.
# This catches numerical instability early before it corrupts the model.
# ---------------------------------------------------------------------------
_DIVERGENCE_THRESHOLD: float = 1e6


def _create_lr_scheduler(
    optimizer: AdamW,
    total_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """Create a cosine learning rate scheduler with linear warmup.

    The schedule has two phases:
    1. Linear warmup: LR increases linearly from 0 to base_lr over warmup_epochs.
    2. Cosine decay: LR decreases from base_lr to min_lr following a cosine curve
       over the remaining epochs.

    Args:
        optimizer: The AdamW optimizer whose learning rate will be scheduled.
        total_epochs: Total number of training epochs (20).
        warmup_epochs: Number of epochs for linear warmup (2).
        min_lr: Minimum learning rate at the end of cosine decay (1e-6).
        base_lr: Base (peak) learning rate after warmup (1e-4).

    Returns:
        A LambdaLR scheduler that adjusts the learning rate each epoch.
    """

    def lr_lambda(current_epoch: int) -> float:
        """Compute the learning rate multiplier for a given epoch.

        Args:
            current_epoch: The current epoch number (0-indexed).

        Returns:
            A float multiplier applied to the base learning rate.
        """
        # Phase 1: Linear warmup from 0 to base_lr
        if current_epoch < warmup_epochs:
            # Linear interpolation: epoch 0 → multiplier ~0, epoch warmup-1 → ~1
            return (current_epoch + 1) / warmup_epochs

        # Phase 2: Cosine decay from base_lr to min_lr
        # Compute progress through the decay phase (0.0 to 1.0)
        decay_epochs = total_epochs - warmup_epochs
        progress = (current_epoch - warmup_epochs) / decay_epochs

        # Cosine decay formula: oscillates from 1.0 down to min_lr/base_lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale so that the minimum multiplier corresponds to min_lr
        min_multiplier = min_lr / base_lr
        return min_multiplier + (1.0 - min_multiplier) * cosine_decay

    # Create and return the LambdaLR scheduler using our custom lambda
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def _compute_validation_loss(
    model: PatchTSTModel,
    masker: PatchMasker,
    reconstruction_head: ReconstructionHead,
    val_loader: MultiDomainDataLoader,
    device: torch.device,
) -> float:
    """Compute average validation loss across all domain validation sets.

    Runs the model in evaluation mode (no dropout, no gradient computation)
    on the validation data and returns the mean masked reconstruction loss.

    Args:
        model: The PatchTST encoder model.
        masker: The patch masking module.
        reconstruction_head: The reconstruction head for loss computation.
        val_loader: MultiDomainDataLoader wrapping validation datasets.
        device: The device (CPU or GPU) to run computations on.

    Returns:
        The average validation loss as a float. Returns 0.0 if no batches.
    """
    # Set model and head to evaluation mode (disables dropout)
    model.eval()
    reconstruction_head.eval()

    total_loss = 0.0
    num_batches = 0

    # Disable gradient computation for validation (saves memory and time)
    with torch.no_grad():
        for context_batch, _target_batch, _domain_name in val_loader:
            # Move batch to the appropriate device (GPU or CPU)
            context_batch = context_batch.to(device)

            # Forward pass through the model to get patch embeddings
            encoder_output = model(context_batch)

            # Apply masking to get masked embeddings and mask indices
            masked_output, _original_patches, mask_indices = masker.mask_patches(encoder_output)

            # Run masked embeddings through the encoder again is NOT needed here.
            # For validation, we compute loss on the encoder output directly:
            # The reconstruction head predicts patch values from encoder output.
            reconstructed = reconstruction_head(encoder_output)

            # Create the boolean mask tensor for loss computation
            # mask_indices shape: (batch, num_masked) — indices of masked patches
            batch_size, num_patches, _ = encoder_output.shape
            mask_bool = torch.zeros(
                batch_size, num_patches, dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                mask_bool[i, mask_indices[i]] = True

            # Create target patches from the original input
            # Unfold the context to get the original patch values
            patches_target = context_batch.unfold(
                dimension=1,
                size=Config.PATCH_LEN,
                step=Config.PATCH_STRIDE,
            )

            # Compute masked reconstruction loss (MSE only on masked positions)
            loss = compute_masked_reconstruction_loss(
                reconstructed, patches_target, mask_bool
            )

            total_loss += loss.item()
            num_batches += 1

    # Restore training mode for subsequent training steps
    model.train()
    reconstruction_head.train()

    # Return average loss, or 0.0 if no validation batches were processed
    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def pretrain(
    model: PatchTSTModel,
    train_datasets: list[TimeSeriesDataset],
    val_datasets: list[TimeSeriesDataset],
    config: type = Config,
    device: Optional[torch.device] = None,
    domain_names: Optional[list[str]] = None,
) -> dict:
    """Full pretraining loop: multi-domain masking with checkpointing.

    This function implements the complete Masked Patch Modeling pretraining
    pipeline. It trains the PatchTST model to reconstruct randomly masked
    patches across three domains using round-robin batching, AdamW optimizer
    with cosine LR schedule, and gradient accumulation.

    Key features:
    - Multi-domain round-robin batching (Energy, Weather, Finance)
    - AdamW optimizer with lr=1e-4, weight_decay=0.01
    - Gradient accumulation over 4 steps (effective batch size = 128)
    - Cosine LR schedule with linear warmup over first 2 epochs
    - Checkpoint saved to Google Drive every epoch
    - Validation loss computed across all domains each epoch
    - NaN/divergence detection with graceful shutdown
    - Step timeout detection (>20 min) with auto-checkpoint

    Args:
        model: The PatchTST encoder model to pretrain.
        train_datasets: List of TimeSeriesDataset instances for training
            (one per domain: Energy, Weather, Finance).
        val_datasets: List of TimeSeriesDataset instances for validation
            (one per domain, same order as train_datasets).
        config: Configuration class with training hyperparameters.
            Defaults to the global Config class.
        device: The torch device to train on. If None, auto-detects GPU/CPU.
        domain_names: Optional list of domain name strings for logging.
            Defaults to ["Energy", "Weather", "Finance"].

    Returns:
        A dictionary containing training history:
        {
            "train_losses": list of per-epoch average training losses,
            "val_losses": list of per-epoch average validation losses,
            "learning_rates": list of per-epoch learning rates,
            "epochs_completed": number of epochs successfully completed,
            "stopped_early": whether training was stopped due to divergence,
        }
    """
    # -----------------------------------------------------------------------
    # Device setup: prefer GPU if available, fall back to CPU
    # -----------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Using device: {device}")

    # Default domain names if not provided
    if domain_names is None:
        domain_names = ["Energy", "Weather", "Finance"]

    # -----------------------------------------------------------------------
    # Move model to the target device
    # -----------------------------------------------------------------------
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Initialize the masking module and reconstruction head
    # The masker randomly selects 40% of patches and replaces them with a
    # learnable mask token. The reconstruction head projects encoder output
    # back to patch space for MSE loss computation.
    # -----------------------------------------------------------------------
    masker = PatchMasker(
        mask_ratio=config.MASK_RATIO,
        d_model=config.D_MODEL,
    ).to(device)

    reconstruction_head = ReconstructionHead(
        d_model=config.D_MODEL,
        patch_len=config.PATCH_LEN,
    ).to(device)

    # -----------------------------------------------------------------------
    # Create multi-domain data loaders for training and validation
    # Round-robin batching ensures balanced exposure to all domains.
    # -----------------------------------------------------------------------
    train_loader = MultiDomainDataLoader(
        datasets=train_datasets,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        domain_names=domain_names,
        shuffle=True,
    )

    val_loader = MultiDomainDataLoader(
        datasets=val_datasets,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        domain_names=domain_names,
        shuffle=False,
    )

    # -----------------------------------------------------------------------
    # Collect all trainable parameters from model, masker, and head
    # All three modules are trained jointly during pretraining.
    # -----------------------------------------------------------------------
    all_parameters = (
        list(model.parameters())
        + list(masker.parameters())
        + list(reconstruction_head.parameters())
    )

    # -----------------------------------------------------------------------
    # AdamW optimizer: lr=1e-4, weight_decay=0.01
    # AdamW decouples weight decay from the gradient update, which is the
    # standard choice for transformer training.
    # -----------------------------------------------------------------------
    optimizer = AdamW(
        all_parameters,
        lr=config.PRETRAIN_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------------
    # Cosine LR schedule with linear warmup over first 2 epochs
    # Warmup prevents large gradient updates early when the model is randomly
    # initialized. Cosine decay gradually reduces the learning rate to min_lr.
    # -----------------------------------------------------------------------
    scheduler = _create_lr_scheduler(
        optimizer=optimizer,
        total_epochs=config.PRETRAIN_EPOCHS,
        warmup_epochs=config.WARMUP_EPOCHS,
        min_lr=config.MIN_LR,
        base_lr=config.PRETRAIN_LR,
    )

    # -----------------------------------------------------------------------
    # Initialize experiment logger (W&B with CSV fallback)
    # If the logger module is not available, we fall back to print statements.
    # -----------------------------------------------------------------------
    logger = None
    if _HAS_LOGGER:
        try:
            logger = ExperimentLogger(config, run_name="pretrain_mpm")
        except Exception as e:
            print(f"[Pretrain] Warning: Could not initialize logger: {e}")
            logger = None

    # Training history tracking and last valid state for divergence recovery
    history = {
        "train_losses": [], "val_losses": [], "learning_rates": [],
        "epochs_completed": 0, "stopped_early": False,
    }
    last_valid_state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": 0, "loss": float("inf"),
    }

    # -----------------------------------------------------------------------
    # Main training loop: iterate over epochs
    # -----------------------------------------------------------------------
    print(f"[Pretrain] Starting training for {config.PRETRAIN_EPOCHS} epochs")
    print(f"[Pretrain] Gradient accumulation: {config.GRADIENT_ACCUMULATION} steps, "
          f"effective batch: {config.PRETRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION}")

    for epoch in range(config.PRETRAIN_EPOCHS):
        epoch_start_time = time.time()

        # Set model and head to training mode (enables dropout)
        model.train()
        masker.train()
        reconstruction_head.train()

        # Accumulate loss for logging
        epoch_total_loss = 0.0
        epoch_num_steps = 0

        # Zero gradients at the start of each epoch
        optimizer.zero_grad()

        # Track gradient accumulation steps within this epoch
        accum_step = 0

        # -------------------------------------------------------------------
        # Iterate over batches from all domains in round-robin order
        # -------------------------------------------------------------------
        for batch_idx, (context_batch, _target_batch, domain_name) in enumerate(
            train_loader
        ):
            step_start_time = time.time()

            # Move batch to device
            context_batch = context_batch.to(device)

            # ---------------------------------------------------------------
            # Forward pass: model → masking → reconstruction → loss
            # ---------------------------------------------------------------

            # Step 1: Get encoder output for the full (unmasked) input
            encoder_output = model(context_batch)

            # Step 2: Apply random masking to the encoder output
            # masked_output has mask token at selected positions
            # mask_indices tells us which positions were masked
            masked_output, _original_patches, mask_indices = masker.mask_patches(encoder_output)

            # Step 3: Run the masked output through the reconstruction head
            # This predicts the original patch values at all positions
            reconstructed = reconstruction_head(masked_output)

            # Step 4: Create boolean mask for loss computation
            batch_size, num_patches, _ = encoder_output.shape
            mask_bool = torch.zeros(
                batch_size, num_patches, dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                mask_bool[i, mask_indices[i]] = True

            # Step 5: Get target patches from the original input
            # Unfold extracts the original patch values for comparison
            patches_target = context_batch.unfold(
                dimension=1,
                size=config.PATCH_LEN,
                step=config.PATCH_STRIDE,
            )

            # Step 6: Compute masked reconstruction loss (MSE on masked only)
            loss = compute_masked_reconstruction_loss(
                reconstructed, patches_target, mask_bool
            )

            # Scale loss by gradient accumulation steps for correct averaging
            scaled_loss = loss / config.GRADIENT_ACCUMULATION

            # ---------------------------------------------------------------
            # Backward pass: compute gradients
            # ---------------------------------------------------------------
            scaled_loss.backward()

            # ---------------------------------------------------------------
            # Check for NaN or divergence in the loss
            # ---------------------------------------------------------------
            current_loss = loss.item()

            if math.isnan(current_loss) or current_loss > _DIVERGENCE_THRESHOLD:
                print(
                    f"\n[Pretrain] ERROR: Training divergence detected at "
                    f"epoch {epoch + 1}, batch {batch_idx + 1}. "
                    f"Loss = {current_loss}"
                )
                print("[Pretrain] Stopping training and saving last valid checkpoint.")

                # Restore last valid state
                model.load_state_dict(last_valid_state["model"])
                optimizer.load_state_dict(last_valid_state["optimizer"])

                # Save checkpoint of last valid state
                if _HAS_COLAB_HELPERS:
                    try:
                        save_checkpoint(
                            model, optimizer,
                            last_valid_state["epoch"],
                            last_valid_state["loss"],
                        )
                        print("[Pretrain] Last valid checkpoint saved.")
                    except Exception as e:
                        print(f"[Pretrain] Warning: Could not save checkpoint: {e}")

                history["stopped_early"] = True
                return history

            # ---------------------------------------------------------------
            # Gradient accumulation: update weights every N steps
            # ---------------------------------------------------------------
            accum_step += 1

            if accum_step % config.GRADIENT_ACCUMULATION == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)

                # Optimizer step: update model weights
                optimizer.step()

                # Zero gradients for the next accumulation cycle
                optimizer.zero_grad()

            # Track epoch loss
            epoch_total_loss += current_loss
            epoch_num_steps += 1

            # ---------------------------------------------------------------
            # Check for step timeout (>20 minutes)
            # ---------------------------------------------------------------
            step_elapsed = time.time() - step_start_time

            if step_elapsed > _STEP_TIMEOUT_SECONDS:
                print(
                    f"\n[Pretrain] WARNING: Step {batch_idx + 1} in epoch "
                    f"{epoch + 1} took {step_elapsed / 60:.1f} minutes "
                    f"(threshold: 20 min). Saving auto-checkpoint."
                )
                # Save an automatic checkpoint to prevent data loss
                if _HAS_COLAB_HELPERS:
                    try:
                        save_checkpoint(
                            model, optimizer, epoch, current_loss
                        )
                    except Exception as e:
                        print(
                            f"[Pretrain] Warning: Auto-checkpoint failed: {e}"
                        )

        # -------------------------------------------------------------------
        # End of epoch: compute metrics and save checkpoint
        # -------------------------------------------------------------------

        # Handle any remaining accumulated gradients at epoch end
        if accum_step % config.GRADIENT_ACCUMULATION != 0:
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Compute average training loss for this epoch
        avg_train_loss = (
            epoch_total_loss / epoch_num_steps if epoch_num_steps > 0 else 0.0
        )

        # Step the learning rate scheduler (once per epoch)
        scheduler.step()

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------------------------
        # Compute validation loss across all domain validation sets
        # -------------------------------------------------------------------
        val_loss = _compute_validation_loss(
            model, masker, reconstruction_head, val_loader, device
        )

        # -------------------------------------------------------------------
        # Update training history
        # -------------------------------------------------------------------
        history["train_losses"].append(avg_train_loss)
        history["val_losses"].append(val_loss)
        history["learning_rates"].append(current_lr)
        history["epochs_completed"] = epoch + 1

        # Update last valid state (for recovery on future divergence)
        last_valid_state = {
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
            "loss": avg_train_loss,
        }

        # -------------------------------------------------------------------
        # Log metrics
        # -------------------------------------------------------------------
        epoch_elapsed = time.time() - epoch_start_time

        print(
            f"[Pretrain] Epoch {epoch + 1}/{config.PRETRAIN_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_elapsed:.1f}s"
        )

        # Log to W&B / CSV if logger is available
        if logger is not None:
            try:
                logger.log_epoch({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch_time_seconds": epoch_elapsed,
                })
            except Exception as e:
                print(f"[Pretrain] Warning: Logging failed: {e}")

        # -------------------------------------------------------------------
        # Save checkpoint to Google Drive at the end of every epoch
        # -------------------------------------------------------------------
        if _HAS_COLAB_HELPERS:
            try:
                checkpoint_path = save_checkpoint(
                    model, optimizer, epoch + 1, avg_train_loss
                )
                print(f"[Pretrain] Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"[Pretrain] Warning: Checkpoint save failed: {e}")
        else:
            print(
                "[Pretrain] Note: colab_helpers not available, "
                "skipping Drive checkpoint."
            )

        # -------------------------------------------------------------------
        # Check VRAM usage if helper is available
        # -------------------------------------------------------------------
        if _HAS_COLAB_HELPERS:
            try:
                check_vram()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Training complete
    # -----------------------------------------------------------------------
    print(f"\n[Pretrain] Training complete! "
          f"Epochs: {history['epochs_completed']}/{config.PRETRAIN_EPOCHS} | "
          f"Final train loss: {history['train_losses'][-1]:.6f} | "
          f"Final val loss: {history['val_losses'][-1]:.6f}")

    # Finish the logger session (close W&B run)
    if logger is not None:
        try:
            logger.finish()
        except Exception:
            pass

    return history
