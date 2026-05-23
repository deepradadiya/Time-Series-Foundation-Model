"""PyTorch Dataset and multi-domain data loader for the Time Series Foundation Model.

This module provides two main classes:
  - TimeSeriesDataset: A PyTorch Dataset that yields (context_window, target) pairs
    from a single domain's normalized time series data. It uses a sliding window
    approach to extract overlapping training samples.
  - MultiDomainDataLoader: A custom iterable that performs round-robin interleaved
    batching across multiple domain datasets (Energy, Weather, Finance), drawing
    one batch from each dataset in rotation until the smallest dataset is exhausted.

Related modules:
    - config.py: Provides CONTEXT_LENGTH (512), FORECAST_HORIZON (96), and
      PRETRAIN_BATCH_SIZE (32) used as defaults.
    - data/preprocess.py: Produces the normalized arrays consumed by TimeSeriesDataset.
    - data/patching.py: Patching is applied downstream by the model's patch embedding
      layer, not within this dataset (we yield raw context windows here).
    - pretraining/train.py: Uses MultiDomainDataLoader for the pretraining loop.
"""

import logging
import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config

# Module-level logger for dataset-related warnings and info messages
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset yielding (context_window, target) pairs for a single domain.

    This dataset extracts sliding windows from a normalized univariate or
    multivariate time series. Each sample consists of:
      - context_window: A 1-D tensor of length CONTEXT_LENGTH (512) used as
        model input for pretraining or forecasting.
      - target: A 1-D tensor of length FORECAST_HORIZON (96) representing the
        ground-truth future values immediately following the context window.

    For multivariate data, each channel is treated independently (channel-
    independent design), so the dataset iterates over all channels.

    Attributes:
        data: The normalized time series array (time_steps,) or (time_steps, channels).
        context_length: Number of input time steps per sample (default 512).
        forecast_horizon: Number of target time steps per sample (default 96).
        samples: List of (channel_index, start_index) tuples identifying each sample.
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int = Config.CONTEXT_LENGTH,
        forecast_horizon: int = Config.FORECAST_HORIZON,
    ) -> None:
        """Initialize the dataset by computing all valid sliding window positions.

        Parameters:
            data: A numpy array of shape (time_steps,) for univariate data or
                  (time_steps, num_channels) for multivariate data. Should be
                  normalized (z-score) before passing here.
            context_length: Number of time steps in the input context window.
                            Defaults to Config.CONTEXT_LENGTH (512).
            forecast_horizon: Number of time steps in the prediction target.
                              Defaults to Config.FORECAST_HORIZON (96).
        """
        # Store configuration parameters for use in __getitem__
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        # Total window size needed for one sample (context + target)
        self.window_size = context_length + forecast_horizon

        # Ensure data is 2D: (time_steps, num_channels) for uniform handling
        if data.ndim == 1:
            # Reshape univariate series to (time_steps, 1)
            self.data = data.reshape(-1, 1)
        else:
            self.data = data

        # Number of time steps and channels in the dataset
        self.num_timesteps = self.data.shape[0]
        self.num_channels = self.data.shape[1]

        # Pre-compute all valid (channel, start_index) pairs for fast indexing
        # A valid start index allows extracting a full window (context + target)
        self.samples: list[tuple[int, int]] = []

        # Iterate over each channel independently (channel-independent design)
        for ch in range(self.num_channels):
            # Number of valid starting positions for this channel
            # The last valid start is where start + window_size <= num_timesteps
            num_valid_starts = self.num_timesteps - self.window_size + 1

            # Only add samples if the series is long enough for at least one window
            if num_valid_starts > 0:
                for start_idx in range(num_valid_starts):
                    self.samples.append((ch, start_idx))
            else:
                # Log a warning if a channel is too short for even one sample
                logger.warning(
                    "Channel %d has %d time steps, which is less than the "
                    "required window size of %d (context=%d + horizon=%d). "
                    "No samples will be generated for this channel.",
                    ch,
                    self.num_timesteps,
                    self.window_size,
                    context_length,
                    forecast_horizon,
                )

    def __len__(self) -> int:
        """Return the total number of (context, target) samples in the dataset.

        Returns:
            The number of valid sliding window positions across all channels.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (context_window, target) pair by index.

        Parameters:
            idx: Integer index into the pre-computed samples list.

        Returns:
            A tuple of two float32 tensors:
              - context_window: Shape (context_length,) — the input sequence.
              - target: Shape (forecast_horizon,) — the ground-truth future values.
        """
        # Look up which channel and starting position this index corresponds to
        channel_idx, start_idx = self.samples[idx]

        # Extract the context window: [start, start + context_length)
        context_start = start_idx
        context_end = start_idx + self.context_length
        context_window = self.data[context_start:context_end, channel_idx]

        # Extract the target: [start + context_length, start + context_length + horizon)
        target_start = context_end
        target_end = context_end + self.forecast_horizon
        target = self.data[target_start:target_end, channel_idx]

        # Convert numpy arrays to PyTorch float32 tensors
        context_tensor = torch.tensor(context_window, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return context_tensor, target_tensor


class MultiDomainDataLoader:
    """Round-robin interleaved batching across multiple domain datasets.

    This loader draws one batch from each domain dataset in rotation (round-robin),
    cycling through domains until the smallest dataset is fully consumed. This
    ensures balanced exposure to all domains during pretraining, preventing the
    model from overfitting to the largest dataset.

    One "epoch" completes when the smallest dataset has been fully iterated through.
    At that point, iteration stops even if larger datasets have remaining samples.

    Attributes:
        datasets: List of TimeSeriesDataset instances (one per domain).
        batch_size: Number of samples per batch from each domain.
        domain_names: Optional list of domain name strings for logging.
        dataloaders: List of PyTorch DataLoader instances wrapping each dataset.
    """

    def __init__(
        self,
        datasets: list[TimeSeriesDataset],
        batch_size: int = Config.PRETRAIN_BATCH_SIZE,
        domain_names: list[str] | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Initialize the multi-domain loader with one DataLoader per domain.

        Parameters:
            datasets: A list of TimeSeriesDataset instances, one for each domain
                      (e.g., [energy_dataset, weather_dataset, finance_dataset]).
            batch_size: Number of samples per batch drawn from each domain.
                        Defaults to Config.PRETRAIN_BATCH_SIZE (32).
            domain_names: Optional list of string names for each domain (e.g.,
                          ["energy", "weather", "finance"]). Used for logging
                          and returned alongside batches. If None, defaults to
                          ["domain_0", "domain_1", ...].
            shuffle: Whether to shuffle samples within each domain's DataLoader.
                     Defaults to True for training.
            num_workers: Number of worker processes for data loading. Defaults
                         to 0 (main process only) for Colab compatibility.
        """
        # Store the datasets and batch size for reference
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Assign default domain names if not provided
        if domain_names is not None:
            self.domain_names = domain_names
        else:
            self.domain_names = [f"domain_{i}" for i in range(len(datasets))]

        # Create a PyTorch DataLoader for each domain dataset
        # Each DataLoader handles batching and optional shuffling independently
        self.dataloaders: list[DataLoader] = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False,
            )
            for dataset in datasets
        ]

        # Compute the number of batches in the smallest dataset
        # This determines when one "epoch" ends (round-robin stops here)
        self._min_batches = min(len(dl) for dl in self.dataloaders)

        # Log dataset sizes for visibility during training
        for i, dl in enumerate(self.dataloaders):
            logger.info(
                "Domain '%s': %d samples, %d batches (batch_size=%d)",
                self.domain_names[i],
                len(self.datasets[i]),
                len(dl),
                batch_size,
            )
        logger.info(
            "MultiDomainDataLoader: epoch ends after %d batches per domain "
            "(limited by smallest dataset).",
            self._min_batches,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
        """Iterate over batches in round-robin order across all domains.

        Yields batches from each domain in rotation: domain_0 batch, domain_1
        batch, domain_2 batch, domain_0 batch, ... until the smallest dataset
        is exhausted.

        Yields:
            A tuple of (context_batch, target_batch, domain_name) where:
              - context_batch: Tensor of shape (batch_size, context_length)
              - target_batch: Tensor of shape (batch_size, forecast_horizon)
              - domain_name: String identifying which domain this batch came from
        """
        # Create fresh iterators for each domain's DataLoader
        iterators = [iter(dl) for dl in self.dataloaders]

        # Track how many batches we've drawn from each domain
        batches_drawn = 0

        # Round-robin loop: draw one batch from each domain per round
        while batches_drawn < self._min_batches:
            # Cycle through each domain in order
            for domain_idx, iterator in enumerate(iterators):
                # Draw the next batch from this domain's iterator
                context_batch, target_batch = next(iterator)

                # Yield the batch along with the domain name for logging
                yield context_batch, target_batch, self.domain_names[domain_idx]

            # Increment the round counter after completing one full round
            batches_drawn += 1

    def __len__(self) -> int:
        """Return the total number of batches yielded per epoch.

        One epoch yields min_batches * num_domains total batches, since we
        draw one batch from each domain per round for min_batches rounds.

        Returns:
            Total number of batches in one full epoch iteration.
        """
        # Total batches = rounds × number of domains
        return self._min_batches * len(self.datasets)

    @property
    def num_domains(self) -> int:
        """Return the number of domains in this multi-domain loader.

        Returns:
            Integer count of domain datasets.
        """
        return len(self.datasets)

    @property
    def batches_per_domain(self) -> int:
        """Return the number of batches drawn from each domain per epoch.

        This equals the number of batches in the smallest dataset, since
        the epoch ends when the smallest dataset is exhausted.

        Returns:
            Number of batches per domain per epoch.
        """
        return self._min_batches


class DomainMixedDataLoader:
    """Weighted domain sampling with configurable ratios.

    This loader produces batches containing samples from all three domains
    (Energy, Weather, Finance) with controlled proportions. Unlike the
    round-robin MultiDomainDataLoader, this class mixes samples from all
    domains within a single batch according to configurable weight ratios.

    Each batch contains (input_tensor, domain_labels) where:
      - input_tensor: shape (batch_size, context_length) — concatenated samples
        from all domains in the batch.
      - domain_labels: shape (batch_size,) — integer tensor with 0=Energy,
        1=Weather, 2=Finance indicating the domain of each sample.

    When a domain's samples are exhausted within an epoch, the loader resamples
    from that domain with replacement to maintain the target ratio. Samples
    within each domain are shuffled independently at the start of each epoch.

    Attributes:
        datasets: List of TimeSeriesDataset instances (one per domain).
        domain_weights: Dict mapping domain name to sampling weight.
        batch_size: Total number of samples per batch across all domains.
        domain_names: List of domain name strings.
    """

    def __init__(
        self,
        datasets: list[TimeSeriesDataset],
        domain_weights: dict[str, float],
        batch_size: int = 32,
        domain_names: list[str] | None = None,
    ) -> None:
        """Initialize the weighted domain mixed data loader.

        Parameters:
            datasets: A list of TimeSeriesDataset instances, one for each domain
                      (e.g., [energy_dataset, weather_dataset, finance_dataset]).
            domain_weights: Dict mapping domain name to its sampling weight.
                            Weights should sum to 1.0 (e.g., {"energy": 0.4,
                            "weather": 0.3, "finance": 0.3}).
            batch_size: Total number of samples per batch across all domains.
                        Defaults to 32.
            domain_names: Optional list of domain name strings. If None, defaults
                          to ["energy", "weather", "finance"].

        Raises:
            ValueError: If any domain dataset contains zero samples.
        """
        if domain_names is None:
            domain_names = ["energy", "weather", "finance"]

        self.datasets = datasets
        self.domain_weights = domain_weights
        self.batch_size = batch_size
        self.domain_names = domain_names

        # Validate that no domain dataset is empty
        for i, dataset in enumerate(datasets):
            if len(dataset) == 0:
                domain_name = domain_names[i] if i < len(domain_names) else f"domain_{i}"
                raise ValueError(
                    f"Domain '{domain_name}' has zero samples. All domain "
                    f"datasets must contain at least one sample."
                )

        # Compute per-domain sample counts for each batch based on weights
        self._domain_counts = self._compute_domain_counts()

        # Compute total number of batches per epoch based on the largest domain
        # relative to its per-batch count
        max_batches = 0
        for i, dataset in enumerate(datasets):
            domain_batches = math.ceil(len(dataset) / self._domain_counts[i])
            max_batches = max(max_batches, domain_batches)
        self._num_batches = max_batches

        logger.info(
            "DomainMixedDataLoader: batch_size=%d, per-domain counts=%s, "
            "batches_per_epoch=%d",
            batch_size,
            {name: count for name, count in zip(domain_names, self._domain_counts)},
            self._num_batches,
        )

    def _compute_domain_counts(self) -> list[int]:
        """Compute per-domain sample counts for each batch.

        Rounds each domain's proportion to the nearest integer. If the sum
        doesn't equal batch_size due to rounding, assigns the remainder to
        the highest-weighted domain.

        Returns:
            List of integer counts, one per domain, summing to batch_size.
        """
        # Get weights in domain_names order
        weights = []
        for name in self.domain_names:
            weights.append(self.domain_weights.get(name, 0.0))

        # Compute raw counts and round to nearest integer
        raw_counts = [w * self.batch_size for w in weights]
        counts = [round(c) for c in raw_counts]

        # Fix any rounding discrepancy by adjusting the highest-weighted domain
        remainder = self.batch_size - sum(counts)
        if remainder != 0:
            # Find the index of the highest-weighted domain
            max_weight_idx = weights.index(max(weights))
            counts[max_weight_idx] += remainder

        return counts

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over mixed-domain batches for one epoch.

        At the start of each epoch, samples within each domain are shuffled
        independently. When a domain is exhausted, it is resampled with
        replacement to maintain the target ratio.

        Yields:
            Tuples of (input_tensor, domain_labels) where:
              - input_tensor: shape (batch_size, context_length)
              - domain_labels: shape (batch_size,) with integer domain indices
        """
        # Create shuffled index arrays for each domain
        domain_indices = []
        for dataset in self.datasets:
            perm = torch.randperm(len(dataset)).tolist()
            domain_indices.append(perm)

        # Track current position within each domain's shuffled indices
        domain_positions = [0] * len(self.datasets)

        for _ in range(self._num_batches):
            batch_inputs = []
            batch_labels = []

            for domain_idx, (dataset, count) in enumerate(
                zip(self.datasets, self._domain_counts)
            ):
                indices = domain_indices[domain_idx]
                pos = domain_positions[domain_idx]

                for _ in range(count):
                    if pos >= len(indices):
                        # Domain exhausted — resample with replacement
                        sample_idx = torch.randint(0, len(dataset), (1,)).item()
                    else:
                        sample_idx = indices[pos]
                        pos += 1

                    # Get the context window (first element of the tuple)
                    context, _ = dataset[sample_idx]
                    batch_inputs.append(context)
                    batch_labels.append(domain_idx)

                domain_positions[domain_idx] = pos

            # Stack into tensors
            input_tensor = torch.stack(batch_inputs, dim=0)
            domain_labels = torch.tensor(batch_labels, dtype=torch.long)

            # Shuffle the batch so domains are interleaved randomly
            shuffle_perm = torch.randperm(input_tensor.size(0))
            input_tensor = input_tensor[shuffle_perm]
            domain_labels = domain_labels[shuffle_perm]

            yield input_tensor, domain_labels

    def __len__(self) -> int:
        """Return the number of batches per epoch.

        Returns:
            Total number of batches yielded in one full epoch iteration.
        """
        return self._num_batches
