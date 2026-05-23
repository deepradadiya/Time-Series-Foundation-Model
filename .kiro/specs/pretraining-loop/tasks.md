# Implementation Plan: Enhanced Pretraining Loop

## Overview

This plan implements the enhanced pretraining pipeline with domain classification, weighted domain sampling, mixed precision training, W&B logging, step-based checkpointing, early stopping, and post-training model export. Implementation proceeds from config additions → data layer → loss module → masking enhancement → main training loop → export logic, with property tests validating each component.

## Tasks

- [x] 1. Add configuration parameters and set up new module structure
  - [x] 1.1 Add enhanced pretraining config parameters to `config.py`
    - Add DOMAIN_WEIGHTS, DOMAIN_LOSS_WEIGHT, NUM_DOMAINS, LOG_EVERY_N_STEPS, CHECKPOINT_EVERY_N_STEPS, MAX_CHECKPOINTS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, WANDB_PROJECT, HF_REPO_NAME, CHECKPOINT_DIR, GDRIVE_CHECKPOINT_DIR
    - _Requirements: 2.3, 3.2, 6.1, 6.4, 7.1, 9.2_

  - [x] 1.2 Create `pretraining/pretrain_losses.py` with DomainClassificationHead class
    - Implement `__init__` with d_model and num_domains parameters
    - Implement `forward` with global average pooling (dim=1) + linear projection
    - Add input validation raising ValueError if last dimension != d_model
    - Output shape: (batch_size, num_domains) unnormalized logits
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.3 Implement `compute_pretrain_loss` function in `pretraining/pretrain_losses.py`
    - Compute reconstruction loss (MSE on masked positions only) using mask_indices to build boolean mask
    - Compute domain classification loss as F.cross_entropy(domain_logits, domain_labels)
    - Compute total_loss = reconstruction_loss + 0.1 * domain_classification_loss
    - Return dict with keys "reconstruction_loss", "domain_classification_loss", "total_loss"
    - Handle edge case: zero masked patches → return zero-valued reconstruction_loss with requires_grad=True
    - Validate domain_labels range, raise ValueError for invalid labels
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 1.4 Write property tests for DomainClassificationHead (Properties 1-2)
    - **Property 1: Domain Classification Head Computation** — verify output equals classifier(mean(input, dim=1)) for any valid input shape
    - **Property 2: Domain Classification Head Input Validation** — verify ValueError raised for mismatched d_model
    - **Validates: Requirements 1.1, 1.2, 1.5, 1.6**

  - [ ]* 1.5 Write property tests for compute_pretrain_loss (Properties 3-6)
    - **Property 3: Reconstruction Loss Ignores Unmasked Positions** — modifying unmasked positions does not change loss
    - **Property 4: Domain Classification Loss Matches Cross-Entropy** — verify equals F.cross_entropy(logits, labels)
    - **Property 5: Total Loss Decomposition** — verify total_loss == recon + 0.1 * domain_cls
    - **Property 6: Invalid Domain Labels Rejection** — verify ValueError for out-of-range labels
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.7**

- [x] 2. Implement DomainMixedDataLoader
  - [x] 2.1 Add `DomainMixedDataLoader` class to `data/dataset.py`
    - Accept datasets list, domain_weights dict, batch_size, domain_names
    - Implement weighted sampling: Energy 40%, Weather 30%, Finance 30% per batch
    - Round per-domain counts to nearest integer, assign remainder to highest-weighted domain
    - Return (input_tensor, domain_labels) tuples where domain_labels is integer tensor (0=Energy, 1=Weather, 2=Finance)
    - Resample with replacement when a domain is exhausted within an epoch
    - Shuffle within each domain at epoch start
    - Raise ValueError if any domain dataset has zero samples
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 2.2 Write property test for DomainMixedDataLoader (Property 7)
    - **Property 7: Batch Structure and Domain Proportions** — verify batch shapes, domain label values in {0,1,2}, and per-domain counts match configured weights
    - **Validates: Requirements 3.1, 3.2, 3.3**

- [x] 3. Enhance PatchMasker interface
  - [x] 3.1 Update `PatchMasker.mask_patches` in `pretraining/masking.py` to return 3-tuple
    - Return (masked_input, original_patches, mask_indices) instead of (masked_embeddings, mask_indices)
    - original_patches = input.clone().detach() captured before masking
    - Handle edge case: mask_ratio resulting in 0 patches → return input unchanged with empty mask_indices
    - Maintain backward compatibility: mask_indices still sorted ascending
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 3.2 Write property tests for enhanced PatchMasker (Properties 13-14)
    - **Property 13: Enhanced Masker Memory Independence** — verify original_patches shares no memory with masked_input and has requires_grad=False
    - **Property 14: Mask Indices Ordering and Count** — verify count equals round(mask_ratio * num_patches) and indices are ascending in [0, num_patches)
    - **Validates: Requirements 11.2, 11.3**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement the enhanced pretraining loop
  - [x] 5.1 Create `pretraining/pretrain_loop.py` with mixed precision and optimizer setup
    - Implement device detection (CUDA vs CPU), model/head initialization
    - Set up AdamW optimizer (lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    - Set up cosine LR scheduler with linear warmup (2 epochs) and min_lr=1e-6
    - Set up GradScaler (enabled only on CUDA)
    - Set up torch.autocast context (enabled only on CUDA)
    - _Requirements: 4.1, 4.2, 4.5, 5.1, 5.2, 5.3, 5.5_

  - [x] 5.2 Implement the main training step logic in `pretraining/pretrain_loop.py`
    - Forward pass within autocast: encoder → masker → reconstruction head + domain classification head
    - Compute multi-task loss via compute_pretrain_loss
    - Scale loss with GradScaler, backward pass
    - Gradient accumulation (4 microbatches), then unscale + clip (max_norm=1.0) + scaler.step + scaler.update
    - Handle GradScaler overflow: skip optimizer step, log event
    - Track global optimizer step count
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.4_

  - [x] 5.3 Implement W&B logging in `pretraining/pretrain_loop.py`
    - Initialize W&B run with project "time-series-foundation-model" and Config hyperparameters
    - Log metrics every 50 optimizer steps: reconstruction_loss, domain_classification_loss, total_loss, domain_classification_accuracy, learning_rate
    - Fall back to stdout printing if wandb not installed or init fails
    - Log partial interval metrics at epoch end
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 5.4 Implement step-based checkpointing in `pretraining/pretrain_loop.py`
    - Save checkpoint every 500 optimizer steps to Google Drive
    - Include: model_state_dict, optimizer_state_dict, scaler_state_dict, scheduler_state_dict, epoch, global_step, best_val_loss
    - Retain at most 5 checkpoints, delete oldest when exceeded
    - Handle Drive not mounted: log warning, continue training
    - Implement resume from checkpoint: restore all state and continue from next step
    - Handle corrupted checkpoint: log warning, start from scratch
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 5.5 Implement epoch validation and early stopping in `pretraining/pretrain_loop.py`
    - Compute validation loss on all three domain validation sets at epoch end
    - Compute per-domain classification accuracy on validation set
    - Print epoch summary: avg train loss, avg val loss, per-domain reconstruction accuracy
    - Implement early stopping: patience=5, min_delta=1e-4
    - Restore best model state when early stopping triggers
    - Log early stopping message with best epoch number
    - Return history with stopped_early=True and correct epochs_completed
    - _Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 5.6 Write property tests for LR schedule and gradient clipping (Properties 8-9)
    - **Property 8: Learning Rate Schedule Correctness** — verify linear warmup then cosine decay, LR never below 1e-6
    - **Property 9: Gradient Clipping Bound** — verify total gradient norm ≤ 1.0 after clipping
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**

  - [ ]* 5.7 Write property test for checkpoint round-trip (Property 10)
    - **Property 10: Checkpoint Round-Trip Preservation** — save then load checkpoint, verify all state components match
    - **Validates: Requirements 7.4**

  - [ ]* 5.8 Write property tests for early stopping (Properties 11-12)
    - **Property 11: Early Stopping Trigger Condition** — verify triggers after 5 non-improving epochs
    - **Property 12: Early Stopping State Restoration** — verify model params match best epoch after trigger
    - **Validates: Requirements 9.2, 9.3**

  - [ ]* 5.9 Write property test for domain classification accuracy (Property 15)
    - **Property 15: Domain Classification Accuracy Computation** — verify accuracy equals fraction where argmax(logits)==label
    - **Validates: Requirements 8.3**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement post-training model export
  - [x] 7.1 Implement model export logic in `pretraining/pretrain_loop.py`
    - Save final_pretrained_model.pt to Google Drive (encoder + reconstruction head + optimizer state_dict)
    - Fall back to local `checkpoints/final_pretrained_model.pt` if Drive fails
    - Push encoder backbone to HuggingFace Hub as "{username}/patchtst-foundation-pretrained"
    - Handle HF Hub auth/network failure: log warning, continue
    - Save pretraining_log.json to Google Drive (final losses, per-epoch arrays, epochs_completed, stopped_early, per-domain accuracies)
    - Fall back to local `checkpoints/pretraining_log.json` if Drive fails
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 7.2 Write unit tests for model export
    - Test Google Drive save success and fallback to local
    - Test HuggingFace Hub push failure handling
    - Test pretraining_log.json content structure
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 8. Integration wiring and final validation
  - [x] 8.1 Wire all components together in `pretrain_enhanced` function
    - Ensure DomainMixedDataLoader feeds into training loop
    - Ensure enhanced PatchMasker 3-tuple return is consumed correctly
    - Ensure DomainClassificationHead receives encoder_output
    - Ensure compute_pretrain_loss receives all required inputs
    - Verify existing `pretrain()` in train.py still works unchanged
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1_

  - [ ]* 8.2 Write integration tests for the full enhanced pretraining loop
    - Test 2-epoch training with small synthetic data
    - Verify loss decreases over steps
    - Verify checkpoints saved at correct intervals
    - Verify early stopping triggers with non-improving losses
    - Verify mixed precision on CUDA (skip on CPU-only)
    - _Requirements: 4.1, 5.1, 6.1, 7.1, 8.1, 9.1_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The existing `pretraining/train.py` pretrain() function is preserved for backward compatibility
- Python is used throughout (matching the existing codebase and design document)
- Hypothesis library is used for property-based tests (consistent with existing `tests/properties/`)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.1", "3.1"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.2", "3.2"] },
    { "id": 3, "tasks": ["5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "5.5"] },
    { "id": 5, "tasks": ["5.6", "5.7", "5.8", "5.9"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["7.2", "8.1"] },
    { "id": 8, "tasks": ["8.2"] }
  ]
}
```
