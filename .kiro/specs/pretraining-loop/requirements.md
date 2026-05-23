# Requirements Document

## Introduction

This feature enhances the existing pretraining pipeline for the PatchTST Time Series Foundation Model by adding domain classification as an auxiliary task, weighted domain sampling, mixed precision training, Weights & Biases logging, early stopping, HuggingFace Hub integration, and step-based checkpointing. The pretraining uses Masked Patch Modeling (MPM) across three domains (Energy, Weather, Finance) with a multi-task loss combining reconstruction and domain classification.

## Glossary

- **Pretrain_Loop**: The main training orchestration module (`pretraining/pretrain_loop.py`) that executes multi-domain masked patch modeling with auxiliary domain classification.
- **Domain_Classification_Head**: A small neural network head that predicts which domain (Energy, Weather, Finance) a sample originated from, using the encoder output.
- **Pretrain_Losses**: The loss computation module (`pretraining/pretrain_losses.py`) that combines masked reconstruction loss with domain classification loss.
- **DomainMixedDataLoader**: A data loader that samples batches containing samples from all three domains with configurable per-domain weight ratios (Energy 40%, Weather 30%, Finance 30%).
- **PatchMasker**: The existing masking module (`pretraining/masking.py`) that replaces a fraction of patch embeddings with a learnable mask token.
- **Reconstruction_Loss**: Mean Squared Error computed only on masked patch positions.
- **Domain_Classification_Loss**: Cross-entropy loss on domain label predictions from the Domain_Classification_Head.
- **Early_Stopping**: A mechanism that halts training when validation loss does not improve for a specified number of consecutive epochs (patience).
- **Pretraining_Log**: A JSON file (`pretraining_log.json`) recording final metrics, loss curves, and per-domain accuracies after pretraining completes.
- **Mixed_Precision**: Training with float16 forward passes via `torch.autocast` while maintaining float32 master weights for numerical stability.

## Requirements

### Requirement 1: Domain Classification Head

**User Story:** As a researcher, I want the model to learn domain-aware representations during pretraining, so that the encoder captures both temporal patterns and domain-specific characteristics.

#### Acceptance Criteria

1. THE Domain_Classification_Head SHALL accept encoder output of shape (batch_size, num_patches, d_model) and produce domain logits of shape (batch_size, num_domains).
2. WHEN encoder output is provided, THE Domain_Classification_Head SHALL apply global average pooling across the patch dimension (dimension 1), reducing (batch_size, num_patches, d_model) to (batch_size, d_model), before passing the result to the classification layer.
3. THE Domain_Classification_Head SHALL output unnormalized logits (no softmax or sigmoid activation) for exactly 3 classes corresponding to Energy (index 0), Weather (index 1), and Finance (index 2) domains.
4. THE Domain_Classification_Head SHALL consist of a single linear projection layer mapping d_model dimensions to num_domains (3) output dimensions.
5. IF the input tensor's last dimension does not equal d_model, THEN THE Domain_Classification_Head SHALL raise a descriptive error indicating the expected and received dimensions.
6. WHEN domain logits are produced during pretraining, THE Domain_Classification_Head SHALL output logits compatible with cross-entropy loss computation (one logit per domain class, with the predicted domain being the argmax of the logits).

### Requirement 2: Multi-Task Pretraining Loss

**User Story:** As a researcher, I want to combine reconstruction loss with domain classification loss, so that the model learns both patch-level reconstruction and domain-level discrimination simultaneously.

#### Acceptance Criteria

1. THE Pretrain_Losses SHALL compute Reconstruction_Loss as MSE only on masked patch positions, excluding visible patches from the loss computation.
2. THE Pretrain_Losses SHALL compute Domain_Classification_Loss as cross-entropy between predicted domain logits of shape (batch_size, num_domains) and true domain labels of shape (batch_size,) containing integer class indices in the range [0, num_domains - 1], where num_domains equals the number of pretraining domains (3: Energy, Weather, Finance).
3. THE Pretrain_Losses SHALL compute total loss as: Reconstruction_Loss + 0.1 * Domain_Classification_Loss.
4. WHEN no patches are masked in a batch, THE Pretrain_Losses SHALL return a zero-valued Reconstruction_Loss as a scalar tensor with requires_grad=True so that backpropagation through the computation graph remains valid.
5. THE Pretrain_Losses SHALL return a dictionary with keys "reconstruction_loss", "domain_classification_loss", and "total_loss", where each value is a scalar tensor that retains its computation graph for gradient computation.
6. THE Pretrain_Losses SHALL compute Domain_Classification_Loss from the mean-pooled encoder output across all patch positions (both masked and visible) in each sample.
7. IF the domain labels tensor contains values outside the range [0, num_domains - 1], THEN THE Pretrain_Losses SHALL raise a ValueError indicating the invalid label range.

### Requirement 3: Weighted Domain Mixed Data Loading

**User Story:** As a researcher, I want batches to contain samples from all three domains with controlled proportions, so that the model receives balanced multi-domain exposure with emphasis on energy data.

#### Acceptance Criteria

1. THE DomainMixedDataLoader SHALL produce batches of size equal to PRETRAIN_BATCH_SIZE (32) containing samples from all three domains (Energy, Weather, Finance) randomly interleaved within each batch.
2. THE DomainMixedDataLoader SHALL sample domains with weights: Energy 40%, Weather 30%, Finance 30% of each batch, where the per-domain sample count is rounded to the nearest integer and any remainder is assigned to the highest-weighted domain to maintain a total of PRETRAIN_BATCH_SIZE samples.
3. THE DomainMixedDataLoader SHALL return batch tuples of (input_tensor, domain_labels) where input_tensor has shape (batch_size, context_length) and domain_labels is an integer tensor of shape (batch_size,) with values 0 for Energy, 1 for Weather, and 2 for Finance.
4. WHEN a domain dataset is exhausted within an epoch, THE DomainMixedDataLoader SHALL resample from that domain with replacement to maintain the target ratio for the remainder of the epoch.
5. THE DomainMixedDataLoader SHALL shuffle samples within each domain independently at the start of each epoch using a new random permutation.
6. IF a domain dataset contains zero samples, THEN THE DomainMixedDataLoader SHALL raise a ValueError indicating which domain has no available data.

### Requirement 4: Mixed Precision Training

**User Story:** As a researcher training on a T4 GPU, I want to use float16 mixed precision, so that training runs faster and uses less GPU memory.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL execute forward passes (encoder, masking, reconstruction head, and loss computation) within a `torch.autocast("cuda", dtype=torch.float16)` context, while maintaining model parameters in float32 as master weights.
2. THE Pretrain_Loop SHALL use `torch.amp.GradScaler` to scale the loss before calling `.backward()`, invoke `scaler.unscale_(optimizer)` before gradient clipping, call `scaler.step(optimizer)` in place of `optimizer.step()`, and call `scaler.update()` after each optimizer step.
3. IF gradient accumulation is active, THEN THE Pretrain_Loop SHALL scale each per-microbatch loss with the GradScaler before the backward pass and defer `scaler.step(optimizer)` and `scaler.update()` until the accumulation boundary (every 4 microbatches).
4. IF a gradient overflow is detected by the GradScaler (indicated by `scaler.step` skipping the optimizer update), THEN THE Pretrain_Loop SHALL skip the optimizer step for that accumulation cycle, log the skip event including the current epoch and batch index, and call `scaler.update()` to adjust the scale factor.
5. WHILE running on a CPU-only device, THE Pretrain_Loop SHALL disable both the `torch.autocast` context and the `GradScaler` (setting `enabled=False` on both), and train entirely in float32 mode.

### Requirement 5: Optimizer and Learning Rate Schedule

**User Story:** As a researcher, I want AdamW with cosine annealing and warmup, so that training converges smoothly with proper learning rate dynamics.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL use AdamW optimizer with learning rate 1e-4, weight_decay 0.01, and default betas (0.9, 0.999).
2. THE Pretrain_Loop SHALL apply linear warmup over the first 2 epochs of training, increasing the learning rate linearly from 0 to 1e-4 at each scheduler step.
3. WHEN warmup completes, THE Pretrain_Loop SHALL decay the learning rate following a cosine annealing schedule from 1e-4 to a minimum learning rate of 1e-6 over the remaining epochs.
4. THE Pretrain_Loop SHALL apply gradient clipping with max_norm=1.0 before each optimizer step.
5. IF the computed learning rate falls below 1e-6 at any point during cosine decay, THEN THE Pretrain_Loop SHALL clamp the learning rate to 1e-6.

### Requirement 6: Weights & Biases Logging

**User Story:** As a researcher, I want detailed step-level metrics logged to W&B, so that I can monitor training progress and diagnose issues in real time.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL log metrics to Weights & Biases every 50 optimizer steps, where one optimizer step equals one weight update after gradient accumulation.
2. THE Pretrain_Loop SHALL log the following metrics at each logging interval: reconstruction_loss, domain_classification_loss, total_loss, domain_classification_accuracy, and current learning_rate.
3. IF the `wandb` package is not installed or W&B initialization fails, THEN THE Pretrain_Loop SHALL fall back to printing each metric name and its numeric value to stdout at the same logging interval, without raising an error or interrupting training.
4. THE Pretrain_Loop SHALL initialize a W&B run with the project name "time-series-foundation-model" and all hyperparameters defined in the Config class at the start of training, before the first optimizer step.
5. WHEN the final optimizer step of an epoch does not align with the 50-step logging interval, THE Pretrain_Loop SHALL log the accumulated metrics for that partial interval at epoch end.

### Requirement 7: Step-Based Checkpointing

**User Story:** As a researcher training on Colab with potential disconnections, I want frequent checkpoints saved to Google Drive, so that I can resume training without losing significant progress.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL save a checkpoint to Google Drive every 500 optimizer steps (where one optimizer step equals one weight update after gradient accumulation), retaining at most 5 checkpoint files and deleting the oldest when the limit is exceeded.
2. THE Pretrain_Loop SHALL include in each checkpoint: model state_dict, optimizer state_dict, GradScaler state_dict, learning rate scheduler state_dict, current epoch, current global optimizer step count, and the best validation loss observed so far.
3. IF Google Drive is not mounted or the save operation fails, THEN THE Pretrain_Loop SHALL log a warning message indicating the failure reason and continue training without interruption.
4. WHEN training resumes from a saved checkpoint, THE Pretrain_Loop SHALL restore model state_dict, optimizer state_dict, GradScaler state_dict, learning rate scheduler state_dict, current epoch, current global step count, and best validation loss, and continue training from the next step after the checkpointed step.
5. IF a checkpoint file is corrupted or incompatible on load, THEN THE Pretrain_Loop SHALL log a warning indicating the load failure and start training from scratch.

### Requirement 8: Epoch Validation and Summary

**User Story:** As a researcher, I want per-epoch validation with domain-level breakdown, so that I can assess model performance across all domains.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL compute validation loss on held-out validation sets from all three domains at the end of each epoch.
2. THE Pretrain_Loop SHALL print an epoch summary containing: average train loss, average validation loss, and reconstruction accuracy per domain.
3. THE Pretrain_Loop SHALL compute domain classification accuracy on the validation set as the fraction of correctly predicted domain labels.

### Requirement 9: Early Stopping

**User Story:** As a researcher, I want training to stop automatically when the model stops improving, so that I avoid wasting compute on overfitting.

#### Acceptance Criteria

1. THE Pretrain_Loop SHALL initialize the best validation loss to positive infinity before the first epoch and update it whenever a lower validation loss is observed.
2. IF validation loss does not decrease by more than 1e-4 compared to the best validation loss for 5 consecutive epochs, THEN THE Pretrain_Loop SHALL stop training early.
3. WHEN early stopping is triggered, THE Pretrain_Loop SHALL restore all jointly-trained parameters (model, masker, and reconstruction head) to their states from the epoch with the best validation loss.
4. WHEN early stopping is triggered, THE Pretrain_Loop SHALL log a message indicating early stopping was triggered and the epoch number at which the best validation loss was observed.
5. WHEN early stopping is triggered, THE Pretrain_Loop SHALL return the training history with the "stopped_early" field set to True and "epochs_completed" set to the epoch at which training was halted.

### Requirement 10: Post-Training Model Export

**User Story:** As a researcher, I want the pretrained model saved and published after training, so that I can reuse it for downstream fine-tuning and share it with the community.

#### Acceptance Criteria

1. WHEN pretraining completes (either by finishing all epochs or early stopping), THE Pretrain_Loop SHALL save the encoder model state_dict, reconstruction head state_dict, and optimizer state_dict to Google Drive in the checkpoint directory using the filename `final_pretrained_model.pt`.
2. IF the Google Drive save in criterion 1 fails due to Drive not being mounted or a write error, THEN THE Pretrain_Loop SHALL log a warning to stdout and save the model to the local filesystem at `checkpoints/final_pretrained_model.pt` as a fallback.
3. WHEN pretraining completes, THE Pretrain_Loop SHALL push the pretrained encoder backbone (PatchTSTModel state_dict only, excluding reconstruction and classification heads) to HuggingFace Hub under the repository name "{username}/patchtst-foundation-pretrained", where "{username}" is read from the authenticated `huggingface_hub` user profile.
4. IF the HuggingFace Hub push fails due to authentication or network issues, THEN THE Pretrain_Loop SHALL log a warning message to stdout indicating the failure reason and continue execution without raising an exception.
5. WHEN pretraining completes, THE Pretrain_Loop SHALL save a `pretraining_log.json` file to the same Google Drive checkpoint directory containing: final train loss (float), final validation loss (float), per-epoch train loss array, per-epoch validation loss array, total epochs completed (integer), whether early stopping was triggered (boolean), and per-domain classification accuracies as a dictionary mapping domain name to accuracy float.
6. IF the `pretraining_log.json` save to Google Drive fails, THEN THE Pretrain_Loop SHALL save the file to the local filesystem at `checkpoints/pretraining_log.json` and log a warning to stdout.

### Requirement 11: Enhanced Patch Masking Interface

**User Story:** As a researcher, I want the masking module to return original patch values alongside masked input, so that loss computation has direct access to reconstruction targets.

#### Acceptance Criteria

1. THE PatchMasker SHALL return a tuple of (masked_input, original_patches, mask_indices) from the mask_patches method, where masked_input has shape (batch_size, num_patches, d_model), original_patches has shape (batch_size, num_patches, d_model), and mask_indices has shape (batch_size, num_masked).
2. THE PatchMasker SHALL preserve original_patches as a cloned copy of the input patch embeddings captured before masking is applied, such that original_patches shares no memory with masked_input and is excluded from the computation graph (no gradient flows through it).
3. THE PatchMasker SHALL ensure mask_indices contains integer indices in ascending order representing the masked positions for each sample in the batch, with num_masked equal to round(mask_ratio * num_patches).
4. IF mask_ratio results in zero patches to mask (num_masked equals 0), THEN THE PatchMasker SHALL return original_patches equal to the input, masked_input equal to the input, and mask_indices as an empty tensor of shape (batch_size, 0).
