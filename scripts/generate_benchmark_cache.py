#!/usr/bin/env python
"""Standalone script to pre-generate the benchmark cache for the Gradio demo.

This script generates `app/benchmark_cache.npz` containing pre-computed
forecasts (ARIMA, Prophet, PatchTST) for 10 evenly-spaced ETTh1 test windows.
The cache enables the Live Benchmark Demo tab to render charts instantly
without computing baselines on-the-fly.

Usage:
    python scripts/generate_benchmark_cache.py

The script will:
1. Load ETTh1 test data (falls back to synthetic data if unavailable)
2. Select 10 evenly-spaced test windows (context_length=512, horizon=96)
3. For each window:
   - Compute ARIMA forecast using evaluation/baselines.py
   - Compute Prophet forecast using evaluation/baselines.py
   - Compute PatchTST forecast using model inference
   - Compute MAE for each method
4. Save all results to app/benchmark_cache.npz

Requirements: 2.1, 2.4
"""

import os
import sys
import time

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    """Generate the benchmark cache by importing and triggering BenchmarkCache."""
    from app.gradio_app import BenchmarkCache

    cache_path = BenchmarkCache.CACHE_PATH

    print("=" * 60)
    print("Benchmark Cache Generation Script")
    print("=" * 60)
    print(f"Cache path: {cache_path}")
    print()

    # Remove existing cache to force recomputation
    if os.path.isfile(cache_path):
        print(f"[INFO] Existing cache found at {cache_path}. Removing to regenerate...")
        os.remove(cache_path)

    # Generate the cache by instantiating BenchmarkCache
    # This triggers _load_or_compute() which will call _compute_and_save()
    # since we removed the existing cache file
    print("[INFO] Starting benchmark cache computation...")
    print("[INFO] This may take several minutes (ARIMA/Prophet fitting is slow).")
    print()

    start_time = time.time()

    try:
        cache = BenchmarkCache()
    except Exception as e:
        print(f"[ERROR] Failed to generate benchmark cache: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Verify the cache was generated successfully
    if not os.path.isfile(cache_path):
        print("[ERROR] Cache file was not created.")
        sys.exit(1)

    num_samples = len(cache.samples)
    if num_samples == 0:
        print("[ERROR] Cache was created but contains no samples.")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"[SUCCESS] Benchmark cache generated successfully!")
    print(f"  Samples: {num_samples}")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Cache file: {cache_path}")
    print()

    # Print summary of MAE scores for each sample
    print("Sample MAE Summary:")
    print("-" * 60)
    print(f"{'Sample':<10} {'ARIMA MAE':<15} {'Prophet MAE':<15} {'PatchTST MAE':<15}")
    print("-" * 60)

    for i, sample in enumerate(cache.samples):
        mae = sample["mae_scores"]
        print(
            f"  {i + 1:<8} "
            f"{mae.get('ARIMA', 0.0):<15.4f} "
            f"{mae.get('Prophet', 0.0):<15.4f} "
            f"{mae.get('PatchTST', 0.0):<15.4f}"
        )

    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
