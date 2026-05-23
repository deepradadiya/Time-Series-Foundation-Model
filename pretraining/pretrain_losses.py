"""Pretraining loss computation for multi-task learning.

This module provides:
- DomainClassificationHead: predicts domain from encoder output via global
  average pooling + linear projection.
- compute_pretrain_loss: combines masked reconstruction loss with domain
  classification loss for multi-task pretraining.

Related modules:
    - pretraining/masking.py — PatchMasker that produces mask_indices
    - pretraining/reconstruction_head.py — reconstructs masked patches
    - config.py — DOMAIN_LOSS_WEIGHT, NUM_DOMAINS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassificationHead(nn.Module):
    """Predicts domain from encoder output via global average pooling + linear.

    Applies global average pooling across the patch dimension (dim=1) to reduce
    (batch_size, num_patches, d_model) to (batch_size, d_model), then projects
    to (batch_size, num_domains) unnormalized logits via a single linear layer.

    Args:
        d_model: Expected last dimension of encoder output. Defaults to 256.
        num_domains: Number of domain classes to predict. Defaults to 3.
    """

    def __init__(self, d_model: int = 256, num_domains: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_domains = num_domains
        self.classifier = nn.Linear(d_model, num_domains)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Compute domain logits from encoder output.

        Args:
            encoder_output: Tensor of shape (batch_size, num_patches, d_model).

        Returns:
            domain_logits: Tensor of shape (batch_size, num_domains) containing
                unnormalized logits (no softmax applied).

        Raises:
            ValueError: If encoder_output's last dimension does not equal d_model.
        """
        if encoder_output.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension to be {self.d_model}, "
                f"but got {encoder_output.shape[-1]}"
            )

        # Global average pooling across patch dimension: (B, num_patches, d_model) -> (B, d_model)
        pooled = encoder_output.mean(dim=1)

        # Linear projection: (B, d_model) -> (B, num_domains)
        domain_logits = self.classifier(pooled)

        return domain_logits


def compute_pretrain_loss(
    reconstructed: torch.Tensor,
    original_patches: torch.Tensor,
    mask_indices: torch.Tensor,
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
    domain_loss_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute multi-task pretraining loss combining reconstruction and domain classification.

    Reconstruction loss is MSE computed only on masked patch positions.
    Domain classification loss is cross-entropy on domain logits vs true labels.
    Total loss combines both with a configurable weight on the domain term.

    Args:
        reconstructed: Predicted patches of shape (B, num_patches, patch_len).
        original_patches: Ground truth patches of shape (B, num_patches, patch_len).
        mask_indices: Integer indices of masked positions, shape (B, num_masked).
        domain_logits: Unnormalized domain predictions, shape (B, num_domains).
        domain_labels: True domain labels, shape (B,) with values in [0, num_domains-1].
        domain_loss_weight: Weight for domain classification loss. Defaults to 0.1.

    Returns:
        Dictionary with keys:
            - "reconstruction_loss": Scalar tensor (MSE on masked positions).
            - "domain_classification_loss": Scalar tensor (cross-entropy).
            - "total_loss": reconstruction_loss + domain_loss_weight * domain_classification_loss.

    Raises:
        ValueError: If domain_labels contains values outside [0, num_domains - 1].
    """
    num_domains = domain_logits.shape[-1]

    # Validate domain labels range
    if domain_labels.numel() > 0:
        min_label = domain_labels.min().item()
        max_label = domain_labels.max().item()
        if min_label < 0 or max_label >= num_domains:
            raise ValueError(
                f"domain_labels contains values outside valid range [0, {num_domains - 1}]. "
                f"Got min={int(min_label)}, max={int(max_label)}."
            )

    # Compute reconstruction loss (MSE on masked positions only)
    batch_size = reconstructed.shape[0]
    num_masked = mask_indices.shape[1] if mask_indices.dim() > 1 else 0

    if num_masked == 0:
        # Edge case: no masked patches — return zero loss with grad support
        reconstruction_loss = torch.zeros(
            1, device=reconstructed.device, dtype=reconstructed.dtype, requires_grad=True
        ).squeeze(0)
    else:
        # Build boolean mask from mask_indices: (B, num_patches)
        num_patches = reconstructed.shape[1]
        mask = torch.zeros(
            batch_size, num_patches, device=reconstructed.device, dtype=torch.bool
        )
        # Scatter True at masked positions
        mask.scatter_(1, mask_indices, True)

        # Select masked positions from both reconstructed and original
        # mask shape: (B, num_patches) -> (B, num_patches, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)

        masked_reconstructed = reconstructed[mask_expanded].view(-1)
        masked_original = original_patches[mask_expanded].view(-1)

        reconstruction_loss = F.mse_loss(masked_reconstructed, masked_original)

    # Compute domain classification loss
    domain_classification_loss = F.cross_entropy(domain_logits, domain_labels)

    # Compute total loss
    total_loss = reconstruction_loss + domain_loss_weight * domain_classification_loss

    return {
        "reconstruction_loss": reconstruction_loss,
        "domain_classification_loss": domain_classification_loss,
        "total_loss": total_loss,
    }
