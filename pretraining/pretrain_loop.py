"""Enhanced pretraining loop with multi-task learning, mixed precision, and logging.

This module implements the production-grade pretraining loop for the PatchTST
Time Series Foundation Model. It orchestrates multi-task training combining
Masked Patch Modeling (reconstruction) with domain classification across
Energy, Weather, and Finance domains.

Features:
    - Multi-task loss (reconstruction + domain classification)
    - Mixed precision training (fp16 on CUDA, fp32 on CPU)
    - Gradient accumulation (4 steps, effective batch size 128)
    - AdamW optimizer with cosine LR schedule and linear warmup
    - Weights & Biases logging every 50 optimizer steps
    - Step-based checkpointing every 500 optimizer steps
    - Early stopping with patience=5
    - HuggingFace Hub model export on completion

Related modules:
    - model/patchtst.py: PatchTSTModel encoder
    - pretraining/pretrain_losses.py: DomainClassificationHead, compute_pretrain_loss
    - pretraining/reconstruction_head.py: ReconstructionHead
    - pretraining/masking.py: PatchMasker
    - data/dataset.py: DomainMixedDataLoader, TimeSeriesDataset
    - config.py: All hyperparameters
"""

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model.patchtst import PatchTSTModel
from pretraining.pretrain_losses import DomainClassificationHead, compute_pretrain_loss
from pretraining.reconstruction_head import ReconstructionHead
from pretraining.masking import PatchMasker
from data.dataset import DomainMixedDataLoader, TimeSeriesDataset

# Optional W&B import — training continues without it
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _create_lr_lambda(warmup_epochs: int, total_epochs: int, min_lr: float, base_lr: float):
    """Create a learning rate lambda for linear warmup + cosine decay.

    During warmup (first warmup_epochs), the LR increases linearly from 0 to base_lr.
    After warmup, the LR decays following a cosine schedule from base_lr to min_lr.
    The LR is never allowed to fall below min_lr.

    Args:
        warmup_epochs: Number of epochs for linear warmup.
        total_epochs: Total number of training epochs.
        min_lr: Minimum learning rate floor.
        base_lr: Base (peak) learning rate.

    Returns:
        A lambda function that takes the current epoch and returns a multiplier
        for the base learning rate.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: scale from 0 to 1 over warmup_epochs
            return epoch / warmup_epochs
        else:
            # Cosine decay from 1.0 to min_lr/base_lr over remaining epochs
            decay_epochs = total_epochs - warmup_epochs
            progress = (epoch - warmup_epochs) / max(decay_epochs, 1)
            # Cosine decay: starts at 1.0, ends at min_lr/base_lr
            min_factor = min_lr / base_lr
            cosine_decay = min_factor + (1.0 - min_factor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            return cosine_decay

    return lr_lambda


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: type = Config,
) -> None:
    """Save a training checkpoint to Google Drive with rotation.

    Saves the full training state to the configured Google Drive checkpoint
    directory. If Drive is not mounted or the save fails, logs a warning and
    continues without interruption.

    Checkpoint rotation: retains at most config.MAX_CHECKPOINTS files. When the
    limit is exceeded, the oldest checkpoint (by step number) is deleted.

    Args:
        model: The model being trained.
        optimizer: The optimizer with current state.
        scaler: The GradScaler with current state.
        scheduler: The LR scheduler with current state.
        epoch: Current epoch number.
        global_step: Current global optimizer step count.
        best_val_loss: Best validation loss observed so far.
        config: Configuration class with checkpoint settings.
    """
    checkpoint_dir = config.GDRIVE_CHECKPOINT_DIR

    # Check if Google Drive is mounted (directory exists or can be created)
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except OSError as e:
        print(
            f"WARNING: Cannot access Google Drive checkpoint directory "
            f"'{checkpoint_dir}': {e}. Skipping checkpoint save."
        )
        return

    # Build checkpoint state dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }

    # Save checkpoint with step-based filename
    filename = f"checkpoint_step_{global_step}.pt"
    filepath = os.path.join(checkpoint_dir, filename)

    try:
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    except OSError as e:
        print(
            f"WARNING: Failed to save checkpoint to '{filepath}': {e}. "
            f"Continuing training."
        )
        return

    # Checkpoint rotation: keep at most MAX_CHECKPOINTS, delete oldest
    try:
        existing_checkpoints = sorted(
            [
                f
                for f in os.listdir(checkpoint_dir)
                if f.startswith("checkpoint_step_") and f.endswith(".pt")
            ],
            key=lambda f: int(f.replace("checkpoint_step_", "").replace(".pt", "")),
        )

        while len(existing_checkpoints) > config.MAX_CHECKPOINTS:
            oldest = existing_checkpoints.pop(0)
            oldest_path = os.path.join(checkpoint_dir, oldest)
            os.remove(oldest_path)
            print(f"Deleted old checkpoint: {oldest_path}")
    except (OSError, ValueError) as e:
        print(f"WARNING: Checkpoint rotation failed: {e}. Continuing training.")


def _init_wandb(config: type) -> bool:
    """Initialize a Weights & Biases run with project name and hyperparameters.

    Attempts to initialize W&B with the project name from config.WANDB_PROJECT
    and logs all Config class attributes as hyperparameters. Returns True if
    initialization succeeds, False otherwise (allowing fallback to stdout).

    Args:
        config: Configuration class with WANDB_PROJECT and hyperparameters.

    Returns:
        True if W&B was initialized successfully, False otherwise.
    """
    if not _WANDB_AVAILABLE:
        print("wandb not installed — falling back to stdout logging.")
        return False

    try:
        # Collect all Config hyperparameters as a dict
        hyperparams = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
        wandb.init(
            project=config.WANDB_PROJECT,
            config=hyperparams,
        )
        return True
    except Exception as e:
        print(f"wandb initialization failed: {e} — falling back to stdout logging.")
        return False


def _log_metrics(metrics_dict: dict, step: int, use_wandb: bool) -> None:
    """Log metrics to W&B or print to stdout as fallback.

    Args:
        metrics_dict: Dictionary of metric names to numeric values.
        step: The current global optimizer step number.
        use_wandb: If True, log to W&B; otherwise print to stdout.
    """
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log(metrics_dict, step=step)
    else:
        metrics_str = ", ".join(
            f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics_dict.items()
        )
        print(f"[Step {step}] {metrics_str}")


def _flush_interval_metrics(
    accumulated_metrics: dict,
    interval_steps: int,
    global_step: int,
    use_wandb: bool,
) -> None:
    """Log accumulated metrics for a partial or full logging interval.

    Computes the average of accumulated metrics over the interval and logs them.
    Resets the accumulator after logging.

    Args:
        accumulated_metrics: Dict mapping metric names to accumulated sums.
        interval_steps: Number of optimizer steps accumulated in this interval.
        global_step: Current global optimizer step for the log entry.
        use_wandb: Whether to use W&B or stdout fallback.
    """
    if interval_steps <= 0:
        return

    avg_metrics = {
        key: value / interval_steps
        for key, value in accumulated_metrics.items()
    }
    _log_metrics(avg_metrics, step=global_step, use_wandb=use_wandb)


def _validate(
    model: nn.Module,
    masker: nn.Module,
    reconstruction_head: nn.Module,
    domain_head: nn.Module,
    val_datasets: list,
    domain_names: list[str],
    device: torch.device,
    config: type,
) -> tuple[float, dict[str, float]]:
    """Compute validation loss and per-domain classification accuracy.

    Iterates over each domain's validation dataset, computing reconstruction
    loss (MSE on masked positions) and domain classification accuracy (fraction
    of correct predictions) per domain.

    Args:
        model: PatchTSTModel encoder.
        masker: PatchMasker instance.
        reconstruction_head: ReconstructionHead instance.
        domain_head: DomainClassificationHead instance.
        val_datasets: List of TimeSeriesDataset instances for validation,
            one per domain.
        domain_names: List of domain name strings.
        device: Device to run validation on.
        config: Configuration class with hyperparameters.

    Returns:
        A tuple of (avg_val_loss, domain_accuracies) where:
            - avg_val_loss: Average validation loss across all domains.
            - domain_accuracies: Dict mapping domain name to classification
              accuracy (fraction of correct predictions).
    """
    model.eval()
    masker.eval()
    reconstruction_head.eval()
    domain_head.eval()

    total_val_loss = 0.0
    total_samples = 0
    domain_accuracies: dict[str, float] = {}

    with torch.no_grad():
        for domain_idx, (dataset, domain_name) in enumerate(
            zip(val_datasets, domain_names)
        ):
            if len(dataset) == 0:
                domain_accuracies[domain_name] = 0.0
                continue

            domain_loss = 0.0
            domain_correct = 0
            domain_total = 0

            # Process validation data in batches
            val_loader = DataLoader(
                dataset,
                batch_size=config.PRETRAIN_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

            for batch in val_loader:
                # TimeSeriesDataset returns (context_window, target)
                input_tensor = batch[0].to(device)
                batch_size = input_tensor.shape[0]

                # Create domain labels for this batch (all same domain)
                domain_labels = torch.full(
                    (batch_size,), domain_idx, dtype=torch.long, device=device
                )

                # Extract raw patches as reconstruction targets
                raw_patches = input_tensor.unfold(
                    dimension=1,
                    size=config.PATCH_LEN,
                    step=config.PATCH_STRIDE,
                )  # (B, num_patches, patch_len)

                # Forward pass
                encoder_output = model(input_tensor)
                masked_input, original_patches, mask_indices = masker.mask_patches(
                    encoder_output
                )
                reconstructed = reconstruction_head(masked_input)
                domain_logits = domain_head(encoder_output)

                # Compute loss
                loss_dict = compute_pretrain_loss(
                    reconstructed=reconstructed,
                    original_patches=raw_patches,
                    mask_indices=mask_indices,
                    domain_logits=domain_logits,
                    domain_labels=domain_labels,
                    domain_loss_weight=config.DOMAIN_LOSS_WEIGHT,
                )

                domain_loss += loss_dict["total_loss"].item() * batch_size
                domain_total += batch_size

                # Domain classification accuracy
                domain_preds = torch.argmax(domain_logits, dim=1)
                domain_correct += (domain_preds == domain_labels).sum().item()

            # Per-domain accuracy
            if domain_total > 0:
                domain_accuracies[domain_name] = domain_correct / domain_total
                total_val_loss += domain_loss
                total_samples += domain_total
            else:
                domain_accuracies[domain_name] = 0.0

    # Average validation loss across all domains
    avg_val_loss = total_val_loss / max(total_samples, 1)

    return avg_val_loss, domain_accuracies


def _export_model(
    model: nn.Module,
    reconstruction_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict,
    config: type = Config,
) -> None:
    """Export pretrained model, push to HuggingFace Hub, and save training log.

    Performs three export operations after pretraining completes:
    1. Saves final_pretrained_model.pt (encoder + reconstruction head + optimizer)
       to Google Drive, falling back to local checkpoints/ if Drive fails.
    2. Pushes encoder backbone to HuggingFace Hub as
       "{username}/patchtst-foundation-pretrained".
    3. Saves pretraining_log.json with training metrics to Google Drive,
       falling back to local checkpoints/ if Drive fails.

    All operations use graceful degradation — failures are logged as warnings
    and execution continues without raising exceptions.

    Args:
        model: The pretrained PatchTSTModel encoder.
        reconstruction_head: The ReconstructionHead used during pretraining.
        optimizer: The optimizer with final state.
        history: Training history dict with losses, metrics, and metadata.
        config: Configuration class with checkpoint directory settings.
    """
    # -------------------------------------------------------------------------
    # 1. Save final_pretrained_model.pt to Google Drive (or local fallback)
    # Contains: encoder state_dict, reconstruction head state_dict, optimizer state_dict
    # -------------------------------------------------------------------------
    model_checkpoint = {
        "model_state_dict": model.state_dict(),
        "reconstruction_head_state_dict": reconstruction_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    filename = "final_pretrained_model.pt"
    saved_to_drive = False

    try:
        gdrive_dir = config.GDRIVE_CHECKPOINT_DIR
        os.makedirs(gdrive_dir, exist_ok=True)
        gdrive_path = os.path.join(gdrive_dir, filename)
        torch.save(model_checkpoint, gdrive_path)
        print(f"Final model saved to Google Drive: {gdrive_path}")
        saved_to_drive = True
    except OSError as e:
        print(
            f"WARNING: Failed to save final model to Google Drive: {e}. "
            f"Falling back to local save."
        )

    if not saved_to_drive:
        try:
            local_dir = config.CHECKPOINT_DIR
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            torch.save(model_checkpoint, local_path)
            print(f"Final model saved locally: {local_path}")
        except OSError as e:
            print(f"WARNING: Failed to save final model locally: {e}")

    # -------------------------------------------------------------------------
    # 2. Push encoder backbone to HuggingFace Hub
    # Only the encoder (PatchTSTModel) state_dict is pushed, excluding
    # reconstruction and classification heads.
    # -------------------------------------------------------------------------
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        # Get the authenticated username
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{config.HF_REPO_NAME}"

        # Save model state_dict to a temporary file for upload
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            torch.save(model.state_dict(), tmp_path)

        try:
            # Create repo if it doesn't exist (no error if it already exists)
            api.create_repo(repo_id=repo_id, exist_ok=True)
            # Upload the model file
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="model.pt",
                repo_id=repo_id,
            )
            print(f"Encoder backbone pushed to HuggingFace Hub: {repo_id}")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"WARNING: HuggingFace Hub push failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Save pretraining_log.json to Google Drive (or local fallback)
    # Contains: final losses, per-epoch arrays, epochs_completed,
    # stopped_early, per-domain accuracies.
    # -------------------------------------------------------------------------
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])

    pretraining_log = {
        "final_train_loss": train_losses[-1] if train_losses else 0.0,
        "final_val_loss": val_losses[-1] if val_losses else 0.0,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_completed": history.get("epochs_completed", 0),
        "stopped_early": history.get("stopped_early", False),
        "domain_accuracies": history.get("domain_accuracies", {}),
    }

    log_filename = "pretraining_log.json"
    log_saved_to_drive = False

    try:
        gdrive_dir = config.GDRIVE_CHECKPOINT_DIR
        os.makedirs(gdrive_dir, exist_ok=True)
        gdrive_log_path = os.path.join(gdrive_dir, log_filename)
        with open(gdrive_log_path, "w") as f:
            json.dump(pretraining_log, f, indent=2)
        print(f"Pretraining log saved to Google Drive: {gdrive_log_path}")
        log_saved_to_drive = True
    except OSError as e:
        print(
            f"WARNING: Failed to save pretraining log to Google Drive: {e}. "
            f"Falling back to local save."
        )

    if not log_saved_to_drive:
        try:
            local_dir = config.CHECKPOINT_DIR
            os.makedirs(local_dir, exist_ok=True)
            local_log_path = os.path.join(local_dir, log_filename)
            with open(local_log_path, "w") as f:
                json.dump(pretraining_log, f, indent=2)
            print(f"Pretraining log saved locally: {local_log_path}")
        except OSError as e:
            print(f"WARNING: Failed to save pretraining log locally: {e}")


def pretrain_enhanced(
    model: PatchTSTModel,
    train_datasets: list[TimeSeriesDataset],
    val_datasets: list[TimeSeriesDataset],
    config: type = Config,
    device: Optional[torch.device] = None,
    domain_names: Optional[list[str]] = None,
    resume_checkpoint: Optional[str] = None,
) -> dict:
    """Full enhanced pretraining loop with multi-task learning and modern ML practices.

    Orchestrates pretraining with:
    - Multi-task loss (reconstruction + domain classification)
    - Mixed precision (fp16 on CUDA, fp32 on CPU)
    - Gradient accumulation (4 steps)
    - AdamW + cosine LR with linear warmup
    - W&B logging every 50 optimizer steps
    - Checkpointing every 500 optimizer steps
    - Early stopping (patience=5)
    - HuggingFace Hub push on completion

    Args:
        model: PatchTSTModel encoder to pretrain.
        train_datasets: List of TimeSeriesDataset instances for training,
            one per domain (Energy, Weather, Finance).
        val_datasets: List of TimeSeriesDataset instances for validation,
            one per domain.
        config: Configuration class with all hyperparameters. Defaults to Config.
        device: Target device for training. If None, auto-detects CUDA or CPU.
        domain_names: List of domain name strings. If None, defaults to
            ["energy", "weather", "finance"].
        resume_checkpoint: Optional path to a checkpoint file to resume from.

    Returns:
        Training history dict with keys:
            - train_losses: list[float] — per-epoch average train loss
            - val_losses: list[float] — per-epoch average validation loss
            - learning_rates: list[float] — per-epoch learning rate
            - domain_accuracies: dict[str, float] — final per-domain accuracy
            - epochs_completed: int — number of epochs completed
            - global_steps_completed: int — total optimizer steps
            - stopped_early: bool — whether early stopping triggered
            - best_epoch: int — epoch with best validation loss
    """
    # -------------------------------------------------------------------------
    # Device Detection
    # Use CUDA if available and no explicit device is provided.
    # -------------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")

    # -------------------------------------------------------------------------
    # Domain Names Setup
    # -------------------------------------------------------------------------
    if domain_names is None:
        domain_names = ["energy", "weather", "finance"]

    # -------------------------------------------------------------------------
    # Model and Heads Initialization
    # Move model and auxiliary heads to the target device.
    # -------------------------------------------------------------------------
    model = model.to(device)

    # Domain classification head: predicts which domain a sample belongs to
    domain_head = DomainClassificationHead(
        d_model=config.D_MODEL,
        num_domains=config.NUM_DOMAINS,
    ).to(device)

    # Reconstruction head: projects encoder output back to patch space
    reconstruction_head = ReconstructionHead(
        d_model=config.D_MODEL,
        patch_len=config.PATCH_LEN,
    ).to(device)

    # Patch masker: randomly masks patches for self-supervised learning
    masker = PatchMasker(
        mask_ratio=config.MASK_RATIO,
        d_model=config.D_MODEL,
    ).to(device)

    # -------------------------------------------------------------------------
    # Optimizer Setup
    # AdamW with lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999)
    # Collects parameters from all trainable components.
    # -------------------------------------------------------------------------
    all_parameters = (
        list(model.parameters())
        + list(domain_head.parameters())
        + list(reconstruction_head.parameters())
        + list(masker.parameters())
    )

    optimizer = torch.optim.AdamW(
        all_parameters,
        lr=config.PRETRAIN_LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # -------------------------------------------------------------------------
    # Learning Rate Scheduler
    # Linear warmup for 2 epochs, then cosine decay to min_lr=1e-6.
    # Uses LambdaLR with a custom lambda that implements both phases.
    # The scheduler steps once per epoch.
    # -------------------------------------------------------------------------
    lr_lambda = _create_lr_lambda(
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.PRETRAIN_EPOCHS,
        min_lr=config.MIN_LR,
        base_lr=config.PRETRAIN_LR,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------------------------------------------------------
    # Mixed Precision Setup
    # GradScaler is enabled only on CUDA for fp16 training.
    # On CPU, both autocast and GradScaler are disabled (train in fp32).
    # -------------------------------------------------------------------------
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------------------------------------------------------------
    # Data Loader Setup
    # DomainMixedDataLoader produces mixed batches with weighted sampling.
    # -------------------------------------------------------------------------
    train_loader = DomainMixedDataLoader(
        datasets=train_datasets,
        domain_weights=config.DOMAIN_WEIGHTS,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        domain_names=domain_names,
    )

    # -------------------------------------------------------------------------
    # Training State Initialization
    # -------------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "domain_accuracies": {},
        "epochs_completed": 0,
        "global_steps_completed": 0,
        "stopped_early": False,
        "best_epoch": 0,
    }

    # Best model state for early stopping restoration
    best_model_state = None
    best_masker_state = None
    best_reconstruction_head_state = None

    # -------------------------------------------------------------------------
    # Resume from Checkpoint (if provided)
    # Restores model, optimizer, scaler, scheduler, and training state.
    # -------------------------------------------------------------------------
    if resume_checkpoint is not None:
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = checkpoint.get("global_step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(
                f"Resumed from checkpoint: epoch={start_epoch}, "
                f"global_step={global_step}, best_val_loss={best_val_loss:.6f}"
            )
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            print(f"WARNING: Failed to load checkpoint '{resume_checkpoint}': {e}")
            print("Starting training from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    # -------------------------------------------------------------------------
    # W&B Initialization
    # Initialize Weights & Biases for metric logging. Falls back to stdout
    # if wandb is not installed or initialization fails.
    # -------------------------------------------------------------------------
    use_wandb = _init_wandb(config)

    # Metric accumulator for interval-based logging (every LOG_EVERY_N_STEPS)
    _accumulated_metrics: dict = {
        "reconstruction_loss": 0.0,
        "domain_classification_loss": 0.0,
        "total_loss": 0.0,
        "domain_classification_accuracy": 0.0,
        "learning_rate": 0.0,
    }
    _interval_steps: int = 0  # steps accumulated since last log

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    accumulation_steps = config.GRADIENT_ACCUMULATION  # 4 microbatches per optimizer step

    # -------------------------------------------------------------------------
    # Main Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, config.PRETRAIN_EPOCHS):
        model.train()
        domain_head.train()
        reconstruction_head.train()
        masker.train()

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_domain_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, (input_tensor, domain_labels) in enumerate(train_loader):
            # -----------------------------------------------------------------
            # Move data to device
            # -----------------------------------------------------------------
            input_tensor = input_tensor.to(device)
            domain_labels = domain_labels.to(device)

            # -----------------------------------------------------------------
            # Extract raw patches as reconstruction targets
            # Raw patches have shape (B, num_patches, patch_len) — the original
            # time series values before embedding. These serve as ground truth
            # for the reconstruction loss.
            # -----------------------------------------------------------------
            raw_patches = input_tensor.unfold(
                dimension=1,
                size=config.PATCH_LEN,
                step=config.PATCH_STRIDE,
            )  # (B, num_patches, patch_len)

            # -----------------------------------------------------------------
            # Forward pass within autocast context (fp16 on CUDA, fp32 on CPU)
            # -----------------------------------------------------------------
            with torch.autocast("cuda", enabled=use_amp):
                # Encoder: (B, context_length) -> (B, num_patches, d_model)
                encoder_output = model(input_tensor)

                # Masker: (B, num_patches, d_model) -> (masked_input, original_patches, mask_indices)
                masked_input, original_patches, mask_indices = masker.mask_patches(
                    encoder_output
                )

                # Reconstruction head: (B, num_patches, d_model) -> (B, num_patches, patch_len)
                reconstructed = reconstruction_head(masked_input)

                # Domain classification head: (B, num_patches, d_model) -> (B, num_domains)
                domain_logits = domain_head(encoder_output)

                # Compute multi-task loss
                # Note: original_patches for reconstruction loss are the raw patches
                # (patch_len space), not the encoder embeddings (d_model space).
                loss_dict = compute_pretrain_loss(
                    reconstructed=reconstructed,
                    original_patches=raw_patches,
                    mask_indices=mask_indices,
                    domain_logits=domain_logits,
                    domain_labels=domain_labels,
                    domain_loss_weight=config.DOMAIN_LOSS_WEIGHT,
                )

                loss = loss_dict["total_loss"]

                # Scale loss by accumulation steps for gradient averaging
                scaled_loss = loss / accumulation_steps

            # -----------------------------------------------------------------
            # Backward pass with GradScaler
            # -----------------------------------------------------------------
            scaler.scale(scaled_loss).backward()

            # Track running losses for epoch summary
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict["reconstruction_loss"].item()
            epoch_domain_loss += loss_dict["domain_classification_loss"].item()
            num_batches += 1

            # -----------------------------------------------------------------
            # Gradient accumulation boundary: every `accumulation_steps` microbatches
            # -----------------------------------------------------------------
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)

                # Clip gradients (max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)

                # Optimizer step via scaler (handles overflow detection)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()

                # Detect GradScaler overflow: if scale decreased, step was skipped
                if new_scale < old_scale:
                    print(
                        f"[Epoch {epoch}, Batch {batch_idx}] "
                        f"GradScaler overflow detected — optimizer step skipped. "
                        f"Scale: {old_scale:.1f} -> {new_scale:.1f}"
                    )
                else:
                    # Only increment global_step when optimizer actually stepped
                    global_step += 1

                # Reset gradients for next accumulation cycle
                optimizer.zero_grad()

                # ---------------------------------------------------------
                # W&B logging every LOG_EVERY_N_STEPS optimizer steps
                # ---------------------------------------------------------
                # Compute domain classification accuracy for this step
                with torch.no_grad():
                    domain_preds = torch.argmax(domain_logits, dim=1)
                    domain_acc = (domain_preds == domain_labels).float().mean().item()

                # Accumulate metrics for interval averaging
                _accumulated_metrics["reconstruction_loss"] += loss_dict[
                    "reconstruction_loss"
                ].item()
                _accumulated_metrics["domain_classification_loss"] += loss_dict[
                    "domain_classification_loss"
                ].item()
                _accumulated_metrics["total_loss"] += loss_dict["total_loss"].item()
                _accumulated_metrics["domain_classification_accuracy"] += domain_acc
                _accumulated_metrics["learning_rate"] += optimizer.param_groups[0]["lr"]
                _interval_steps += 1

                # Log at every LOG_EVERY_N_STEPS interval
                if global_step > 0 and global_step % config.LOG_EVERY_N_STEPS == 0:
                    _flush_interval_metrics(
                        _accumulated_metrics,
                        _interval_steps,
                        global_step,
                        use_wandb,
                    )
                    # Reset accumulators
                    for key in _accumulated_metrics:
                        _accumulated_metrics[key] = 0.0
                    _interval_steps = 0

                # ---------------------------------------------------------
                # Checkpointing every CHECKPOINT_EVERY_N_STEPS optimizer steps
                # ---------------------------------------------------------
                if global_step > 0 and global_step % config.CHECKPOINT_EVERY_N_STEPS == 0:
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=config,
                    )

        # ---------------------------------------------------------------------
        # Handle remaining gradients if last batch didn't align with accumulation
        # ---------------------------------------------------------------------
        if num_batches % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale < old_scale:
                print(
                    f"[Epoch {epoch}, End] "
                    f"GradScaler overflow detected — optimizer step skipped. "
                    f"Scale: {old_scale:.1f} -> {new_scale:.1f}"
                )
            else:
                global_step += 1

            optimizer.zero_grad()

        # ---------------------------------------------------------------------
        # Flush partial interval metrics at epoch end (Requirement 6.5)
        # If the last optimizer step didn't align with LOG_EVERY_N_STEPS,
        # log whatever has accumulated so far.
        # ---------------------------------------------------------------------
        if _interval_steps > 0:
            _flush_interval_metrics(
                _accumulated_metrics,
                _interval_steps,
                global_step,
                use_wandb,
            )
            # Reset accumulators for next epoch
            for key in _accumulated_metrics:
                _accumulated_metrics[key] = 0.0
            _interval_steps = 0

        # ---------------------------------------------------------------------
        # Epoch Summary
        # ---------------------------------------------------------------------
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        history["train_losses"].append(avg_epoch_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}/{config.PRETRAIN_EPOCHS - 1} — "
            f"Train Loss: {avg_epoch_loss:.6f} "
            f"(Recon: {epoch_recon_loss / max(num_batches, 1):.6f}, "
            f"Domain: {epoch_domain_loss / max(num_batches, 1):.6f}) — "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} — "
            f"Global Step: {global_step}"
        )

        # Step the learning rate scheduler (once per epoch)
        scheduler.step()

        # ---------------------------------------------------------------------
        # Epoch-end Validation and Early Stopping
        # ---------------------------------------------------------------------
        val_loss, domain_accuracies = _validate(
            model=model,
            masker=masker,
            reconstruction_head=reconstruction_head,
            domain_head=domain_head,
            val_datasets=val_datasets,
            domain_names=domain_names,
            device=device,
            config=config,
        )

        history["val_losses"].append(val_loss)
        history["domain_accuracies"] = domain_accuracies

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{config.PRETRAIN_EPOCHS}: "
            f"train_loss={avg_epoch_loss:.4f}, val_loss={val_loss:.4f}"
        )
        for dname, acc in domain_accuracies.items():
            print(f"  {dname} accuracy: {acc:.4f}")

        # Early stopping check
        min_delta = config.EARLY_STOPPING_MIN_DELTA
        patience = config.EARLY_STOPPING_PATIENCE

        if val_loss < best_val_loss - min_delta:
            # Improvement found — save best state
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            history["best_epoch"] = best_epoch

            # Deep copy model states for restoration
            best_model_state = copy.deepcopy(model.state_dict())
            best_masker_state = copy.deepcopy(masker.state_dict())
            best_reconstruction_head_state = copy.deepcopy(
                reconstruction_head.state_dict()
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            # Early stopping triggered — restore best model state
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best validation loss was at epoch {best_epoch + 1}."
            )

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                masker.load_state_dict(best_masker_state)
                reconstruction_head.load_state_dict(best_reconstruction_head_state)

            history["stopped_early"] = True
            history["epochs_completed"] = epoch + 1
            history["global_steps_completed"] = global_step
            break

        # Set models back to train mode for next epoch
        model.train()
        domain_head.train()
        reconstruction_head.train()
        masker.train()

    # -------------------------------------------------------------------------
    # Finalize training history
    # -------------------------------------------------------------------------
    if not history["stopped_early"]:
        history["epochs_completed"] = len(history["train_losses"])
        history["global_steps_completed"] = global_step

    # -------------------------------------------------------------------------
    # Post-Training Model Export (Requirement 10)
    # Save final model, push to HuggingFace Hub, and save training log.
    # -------------------------------------------------------------------------
    _export_model(
        model=model,
        reconstruction_head=reconstruction_head,
        optimizer=optimizer,
        history=history,
        config=config,
    )

    # Close W&B run if active
    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception:
            pass

    return history
