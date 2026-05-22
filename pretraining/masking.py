"""Random patch masking module for Masked Patch Modeling pretraining.

This module implements the masking strategy used during self-supervised pretraining.
It randomly selects a fraction (40%) of patch positions in each sample and replaces
them with a learnable mask token. The model then learns to reconstruct the original
patch values at these masked positions, forcing it to develop rich temporal
representations of time series data.

Related modules:
    - config.py provides MASK_RATIO and D_MODEL constants
    - model/patch_embedding.py produces the patch embeddings that this module masks
    - pretraining/reconstruction_head.py reconstructs the original patches at masked positions
    - pretraining/train.py orchestrates the masking within the training loop
"""

import torch
import torch.nn as nn

from config import Config


class PatchMasker(nn.Module):
    """Randomly masks patches and replaces them with a learnable mask token.

    During Masked Patch Modeling (MPM) pretraining, a fixed fraction of patches
    in each sample are randomly selected (without replacement) and replaced with
    a shared learnable mask token. The transformer encoder then processes the
    partially-masked sequence, and a reconstruction head predicts the original
    values of the masked patches.

    The mask token is a single learnable vector of dimension D_MODEL (256) that
    is shared across all masked positions and all samples in the batch. It is
    initialized with small random values and updated via backpropagation during
    training, allowing the model to learn an optimal "placeholder" representation
    for missing patches.

    Parameters:
        mask_ratio (float): Fraction of patches to mask per sample.
            Default is Config.MASK_RATIO = 0.4 (40% of patches).
        d_model (int): Dimension of each patch embedding (and the mask token).
            Default is Config.D_MODEL = 256.

    Example:
        >>> masker = PatchMasker(mask_ratio=0.4, d_model=256)
        >>> embeddings = torch.randn(4, 63, 256)  # batch=4, 63 patches, dim=256
        >>> masked_embeddings, mask_indices = masker.mask_patches(embeddings)
        >>> masked_embeddings.shape  # (4, 63, 256) — same shape, some patches replaced
        >>> mask_indices.shape       # (4, 25) — 25 masked positions per sample (40% of 63)
    """

    def __init__(
        self,
        mask_ratio: float = Config.MASK_RATIO,
        d_model: int = Config.D_MODEL,
    ) -> None:
        """Initialize the patch masker with a learnable mask token.

        Args:
            mask_ratio: Fraction of patches to mask in each sample (0.0 to 1.0).
                A value of 0.4 means 40% of patches will be replaced with the
                mask token during pretraining.
            d_model: Dimension of the patch embeddings and the mask token vector.
                Must match the output dimension of the patch embedding module.
        """
        # Call parent nn.Module constructor to register parameters
        super().__init__()

        # Store the mask ratio for computing how many patches to mask
        self.mask_ratio = mask_ratio

        # Store the embedding dimension for reference
        self.d_model = d_model

        # -----------------------------------------------------------------------
        # Learnable Mask Token
        # This is a single vector of shape (d_model,) = (256,) that replaces
        # every masked patch position. It is wrapped in nn.Parameter so that
        # PyTorch includes it in the model's parameter list and updates it
        # during backpropagation.
        #
        # Initialization: small random values from a normal distribution with
        # standard deviation 0.02, following common transformer initialization
        # practices (similar to BERT's [MASK] token initialization). Starting
        # near zero ensures the mask token doesn't dominate early training.
        # -----------------------------------------------------------------------
        self.mask_token = nn.Parameter(
            torch.randn(d_model) * 0.02
        )

    def mask_patches(
        self, patch_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to a batch of patch embeddings.

        For each sample in the batch, this method:
        1. Computes the number of patches to mask: round(mask_ratio * num_patches)
        2. Randomly selects that many patch positions (uniform, without replacement)
        3. Replaces the selected patch embeddings with the learnable mask token
        4. Returns the masked embeddings and the indices of masked positions

        The masking is performed independently for each sample — different samples
        in the same batch will have different patches masked. This provides diverse
        training signal and prevents the model from memorizing specific mask patterns.

        Args:
            patch_embeddings: Input tensor of shape (batch_size, num_patches, d_model).
                These are the output of the PatchEmbedding module — each patch has
                already been projected to dimension 256 and has positional encoding
                added.

        Returns:
            A tuple of two tensors:
            - masked_embeddings: Tensor of shape (batch_size, num_patches, d_model)
                where selected patches have been replaced with the mask token.
                Unmasked patches remain unchanged.
            - mask_indices: Tensor of shape (batch_size, num_masked) containing the
                integer indices of the masked patch positions for each sample.
                num_masked = round(mask_ratio * num_patches).
        """
        # Extract dimensions from the input tensor
        batch_size, num_patches, d_model = patch_embeddings.shape

        # -----------------------------------------------------------------------
        # Step 1: Compute the number of patches to mask
        # We use round() to get the closest integer to mask_ratio * num_patches.
        # For the default configuration: round(0.4 * 63) = round(25.2) = 25 patches.
        # This ensures a consistent number of masked patches across all samples.
        # -----------------------------------------------------------------------
        num_masked = round(self.mask_ratio * num_patches)

        # -----------------------------------------------------------------------
        # Step 2: Clone the input embeddings to avoid modifying the original tensor
        # We need to preserve the original embeddings for computing the
        # reconstruction loss later (comparing predictions against true values).
        # -----------------------------------------------------------------------
        masked_embeddings = patch_embeddings.clone()

        # -----------------------------------------------------------------------
        # Step 3: Generate random mask indices for each sample in the batch
        # For each sample, we randomly select num_masked positions from the
        # range [0, num_patches) without replacement. This is done using
        # torch.randperm to generate a random permutation of all patch indices,
        # then taking the first num_masked indices.
        #
        # We collect indices for all samples into a list, then stack them into
        # a single tensor of shape (batch_size, num_masked).
        # -----------------------------------------------------------------------
        mask_indices_list = []

        for i in range(batch_size):
            # Generate a random permutation of patch indices [0, 1, ..., num_patches-1]
            # and select the first num_masked indices as the positions to mask
            perm = torch.randperm(num_patches, device=patch_embeddings.device)
            selected_indices = perm[:num_masked]

            # Sort the selected indices for consistent ordering (optional but helps
            # with debugging and reproducibility of mask patterns)
            selected_indices, _ = torch.sort(selected_indices)

            mask_indices_list.append(selected_indices)

        # Stack all per-sample index tensors into a single batch tensor
        # Shape: (batch_size, num_masked)
        mask_indices = torch.stack(mask_indices_list, dim=0)

        # -----------------------------------------------------------------------
        # Step 4: Replace masked positions with the learnable mask token
        # For each sample, we index into the masked_embeddings tensor at the
        # selected positions and overwrite those patch vectors with the shared
        # mask token. The mask token is broadcast across all masked positions.
        # -----------------------------------------------------------------------
        for i in range(batch_size):
            # Replace each masked patch embedding with the mask token
            # mask_indices[i] has shape (num_masked,) — indices of patches to mask
            # self.mask_token has shape (d_model,) — broadcast to fill each position
            masked_embeddings[i, mask_indices[i], :] = self.mask_token

        return masked_embeddings, mask_indices
