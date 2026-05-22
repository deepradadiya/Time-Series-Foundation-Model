"""MSE reconstruction head for Masked Patch Modeling pretraining.

This module implements the reconstruction head that projects transformer encoder
outputs back to the original patch space. During pretraining, the model masks
40% of input patches and learns to reconstruct them. The reconstruction head
takes the encoder's contextualized embeddings (d_model=256) and projects them
back to the patch length (16), then computes MSE loss only over the masked
positions — unmasked patches are ignored in the loss computation.

Related modules:
    - model/patchtst.py provides the encoder that produces (batch, num_patches, d_model)
    - pretraining/masking.py handles random patch masking and mask token replacement
    - pretraining/train.py orchestrates the full pretraining loop using this head
    - config.py supplies D_MODEL and PATCH_LEN hyperparameters
"""

import torch
import torch.nn as nn

from config import Config


class ReconstructionHead(nn.Module):
    """Linear projection from d_model back to patch_len for MSE reconstruction.

    This head is used exclusively during pretraining with Masked Patch Modeling.
    It takes the encoder output embeddings (one per patch position) and projects
    each embedding back to the original patch length. The reconstructed patches
    are then compared against the true (normalized) input patches using MSE loss,
    but only at positions that were masked — visible patches do not contribute
    to the loss.

    Architecture:
        Input:  (batch, num_patches, d_model)  e.g., (B, 63, 256)
        Linear: d_model (256) → patch_len (16)
        Output: (batch, num_patches, patch_len) e.g., (B, 63, 16)

    Args:
        d_model: Dimension of the encoder output embeddings. Default: 256.
        patch_len: Length of each original time series patch. Default: 16.

    Example:
        >>> head = ReconstructionHead(d_model=256, patch_len=16)
        >>> encoder_out = torch.randn(4, 63, 256)
        >>> reconstructed = head(encoder_out)  # shape: (4, 63, 16)
    """

    def __init__(self, d_model: int = Config.D_MODEL, patch_len: int = Config.PATCH_LEN) -> None:
        """Initialize the reconstruction head with a single linear layer.

        Args:
            d_model: Dimension of the input embeddings from the encoder.
                     Must match the encoder's output dimension (default: 256).
            patch_len: Target output dimension — the length of each original
                       time series patch to reconstruct (default: 16).
        """
        super().__init__()

        # Store dimensions for reference and validation
        self.d_model = d_model
        self.patch_len = patch_len

        # -----------------------------------------------------------------------
        # Linear projection layer
        # Maps each patch embedding from d_model (256) dimensions back to
        # patch_len (16) dimensions. This is the inverse of the patch embedding
        # projection — it reconstructs the original time series values for each
        # patch position from the contextualized encoder representations.
        # -----------------------------------------------------------------------
        self.projection = nn.Linear(d_model, patch_len)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Project encoder embeddings back to patch space for reconstruction.

        Takes the full encoder output (all patch positions, both masked and
        unmasked) and projects each embedding to the original patch length.
        The caller is responsible for selecting only masked positions when
        computing the loss.

        Args:
            encoder_output: Tensor of shape (batch_size, num_patches, d_model)
                containing contextualized embeddings from the transformer encoder.
                For standard configuration: (B, 63, 256).

        Returns:
            Reconstructed patches of shape (batch_size, num_patches, patch_len).
            For standard configuration: (B, 63, 16).
            Each output vector represents the model's reconstruction of the
            original time series values within that patch.
        """
        # -----------------------------------------------------------------------
        # Apply the linear projection to each patch embedding independently.
        # nn.Linear operates on the last dimension, so it maps:
        #   (batch, num_patches, d_model) → (batch, num_patches, patch_len)
        # This produces a reconstruction for every patch position.
        # -----------------------------------------------------------------------
        reconstructed = self.projection(encoder_output)

        return reconstructed


def compute_masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss only over masked patch positions.

    This function calculates the Mean Squared Error between predicted and target
    patch values, but only at positions where the mask indicates a patch was
    masked during pretraining. Unmasked (visible) patches are completely ignored
    in the loss computation — the model is only penalized for its ability to
    reconstruct patches it could not see.

    The loss is averaged over all masked elements (across batch, masked patches,
    and patch length dimensions) to produce a single scalar loss value.

    Args:
        predictions: Reconstructed patches from the reconstruction head.
            Shape: (batch_size, num_patches, patch_len), e.g., (B, 63, 16).
        targets: Original (normalized) input patches that serve as ground truth.
            Shape: (batch_size, num_patches, patch_len), e.g., (B, 63, 16).
        mask: Boolean tensor indicating which patches were masked (True = masked).
            Shape: (batch_size, num_patches), e.g., (B, 63).
            True at positions that were masked and should contribute to loss.

    Returns:
        Scalar tensor containing the mean squared error averaged over all
        masked positions. Returns 0.0 if no positions are masked.

    Example:
        >>> predictions = torch.randn(4, 63, 16)
        >>> targets = torch.randn(4, 63, 16)
        >>> mask = torch.zeros(4, 63, dtype=torch.bool)
        >>> mask[:, :25] = True  # 25 out of 63 patches masked (≈40%)
        >>> loss = compute_masked_reconstruction_loss(predictions, targets, mask)
        >>> loss.shape  # scalar
        torch.Size([])
    """
    # -----------------------------------------------------------------------
    # Handle edge case: if no patches are masked, return zero loss.
    # This prevents division by zero and is a valid degenerate case.
    # -----------------------------------------------------------------------
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # -----------------------------------------------------------------------
    # Expand the mask to cover the patch_len dimension.
    # mask shape: (batch, num_patches) → (batch, num_patches, 1)
    # After unsqueeze, broadcasting will apply it across all patch_len values.
    # This selects entire patches (all 16 values) at masked positions.
    # -----------------------------------------------------------------------
    mask_expanded = mask.unsqueeze(-1)  # (batch, num_patches, 1)

    # -----------------------------------------------------------------------
    # Compute element-wise squared error between predictions and targets.
    # Shape: (batch, num_patches, patch_len)
    # -----------------------------------------------------------------------
    squared_error = (predictions - targets) ** 2

    # -----------------------------------------------------------------------
    # Apply the mask: zero out squared errors at unmasked positions.
    # Only masked positions contribute to the loss.
    # mask_expanded broadcasts: (batch, num_patches, 1) * (batch, num_patches, patch_len)
    # -----------------------------------------------------------------------
    masked_squared_error = squared_error * mask_expanded.float()

    # -----------------------------------------------------------------------
    # Compute mean over all masked elements.
    # Total masked elements = number of True values in mask * patch_len
    # This gives us the average MSE across all reconstructed values at
    # masked positions, which is the standard masked reconstruction loss.
    # -----------------------------------------------------------------------
    num_masked_elements = mask.sum().float() * predictions.shape[-1]
    loss = masked_squared_error.sum() / num_masked_elements

    return loss
