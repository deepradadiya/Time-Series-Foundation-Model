"""Evaluation metrics for time series probabilistic forecasting.

This module implements four forecast quality metrics used to evaluate the
PatchTST foundation model on the ETTh1 benchmark dataset:
  - MAE (Mean Absolute Error) — average absolute deviation of point forecasts
  - MSE (Mean Squared Error) — average squared deviation of point forecasts
  - MASE (Mean Absolute Scaled Error) — MAE scaled by a seasonal naive baseline
  - CRPS (Continuous Ranked Probability Score) — proper scoring rule for
    probabilistic forecasts approximated from quantile predictions

Point metrics (MAE, MSE, MASE) use the P50 (median) quantile as the point
forecast. CRPS uses all quantile levels (P10, P50, P90) to assess the full
predictive distribution.

Related modules:
    - forecasting/inference.py produces the quantile predictions consumed here
    - evaluation/evaluate.py orchestrates calling these metrics on ETTh1 test data
    - config.py defines QUANTILES = [0.1, 0.5, 0.9] and FORECAST_HORIZON = 96
"""

import numpy as np


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error between predictions and targets.

    Formula: MAE = (1/N) * Σ|predictions_i - targets_i|
    Plain language: Average of the absolute differences between predicted and
    actual values. Measures typical forecast error magnitude without regard to
    direction. Always >= 0; equals 0 only when predictions perfectly match targets.
    Deterministic — identical inputs always produce identical output.

    Parameters
    ----------
    predictions : np.ndarray
        Point forecast values (typically P50 quantile). Shape can be any
        broadcastable shape, but must match targets.
    targets : np.ndarray
        Actual observed values with the same shape as predictions.

    Returns
    -------
    float
        The mean absolute error averaged over all elements. Always >= 0.
    """
    # Convert inputs to numpy arrays for safety (handles lists, tensors, etc.)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # MAE = mean(|predictions - targets|)
    # Compute element-wise absolute differences and take the mean
    absolute_errors = np.abs(predictions - targets)
    return float(np.mean(absolute_errors))


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Squared Error between predictions and targets.

    Formula: MSE = (1/N) * Σ(predictions_i - targets_i)²
    Plain language: Average of the squared differences between predicted and
    actual values. Penalizes larger errors more heavily than MAE due to squaring.
    Always >= 0; equals 0 only when predictions perfectly match targets.
    Deterministic — identical inputs always produce identical output.

    Parameters
    ----------
    predictions : np.ndarray
        Point forecast values (typically P50 quantile). Must have the
        same shape as targets.
    targets : np.ndarray
        Actual observed values with the same shape as predictions.

    Returns
    -------
    float
        The mean squared error averaged over all elements. Always >= 0.
    """
    # Convert inputs to numpy arrays for consistent computation
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # MSE = mean((predictions - targets)²)
    # Square the differences and average across all forecast positions
    squared_errors = (predictions - targets) ** 2
    return float(np.mean(squared_errors))


def mase(
    predictions: np.ndarray,
    targets: np.ndarray,
    seasonal_period: int = 24,
) -> float:
    """Compute Mean Absolute Scaled Error using a seasonal naive baseline.

    Formula: MASE = MAE(predictions, targets) / naive_MAE
             where naive_MAE = mean(|y_t - y_{t-m}|) for t = m, ..., T-1
             and m = seasonal_period (default 24 for hourly/daily seasonality).
    Plain language: Ratio of the model's MAE to the MAE of a seasonal naive
    forecast. A MASE < 1 means the model outperforms the seasonal naive baseline;
    MASE > 1 means it performs worse. Returns float("inf") when the naive baseline
    MAE is zero (perfectly periodic series). Deterministic.

    The seasonal period defaults to 24 (hourly data with daily seasonality,
    matching ETTh1).

    Parameters
    ----------
    predictions : np.ndarray
        Point forecast values (typically P50 quantile). Shape: (n_samples,)
        or (n_windows, horizon) — will be flattened for computation.
    targets : np.ndarray
        Actual observed values with the same shape as predictions.
    seasonal_period : int, optional
        Number of time steps in one seasonal cycle. Defaults to 24 (hourly
        data with daily seasonality as in ETTh1).

    Returns
    -------
    float
        The MASE value. Returns float("inf") if the naive baseline MAE is zero
        (constant series with exact seasonal repetition).
    """
    # Flatten arrays so we can compute the seasonal naive error sequentially
    predictions = np.asarray(predictions, dtype=np.float64).flatten()
    targets = np.asarray(targets, dtype=np.float64).flatten()

    # Numerator: MAE of the model's predictions = mean(|pred - target|)
    numerator = np.mean(np.abs(predictions - targets))

    # Denominator (naive_MAE): MAE of the seasonal naive forecast on targets
    # The seasonal naive forecast at time t is simply the value at t - seasonal_period
    # naive_MAE = mean(|y_t - y_{t-seasonal_period}|) for t = seasonal_period..T-1
    naive_errors = np.abs(targets[seasonal_period:] - targets[:-seasonal_period])

    # Guard against division by zero — return inf when naive baseline MAE is zero
    # (happens if the series is perfectly periodic with period = seasonal_period)
    denominator = np.mean(naive_errors)
    if denominator == 0.0:
        return float("inf")

    # MASE = MAE / naive_MAE
    return float(numerator / denominator)


def crps_quantile(
    q_predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float],
) -> float:
    """Compute CRPS approximated from quantile forecasts.

    Formula: CRPS ≈ (2/K) * Σ_{k=1}^{K} mean(pinball_loss(q_k, y, τ_k))
             where K = number of quantiles (default 3 for P10/P50/P90),
             and pinball_loss(q, y, τ) = τ * max(y - q, 0) + (1 - τ) * max(q - y, 0)
    Plain language: A proper scoring rule for probabilistic forecasts. Measures how
    well the predicted quantiles (P10, P50, P90) capture the true distribution of
    outcomes. Lower CRPS indicates better-calibrated prediction intervals. The factor
    of (2/K) scales the sum of pinball losses to approximate the full CRPS integral.
    Deterministic — identical inputs always produce identical output.

    Default quantile levels: [0.1, 0.5, 0.9] (P10/P50/P90).

    Parameters
    ----------
    q_predictions : np.ndarray
        Quantile predictions with shape (..., num_quantiles). The last
        dimension corresponds to the quantile levels in the same order
        as the `quantiles` parameter. For ETTh1 evaluation this is
        typically (n_windows, horizon, 3) for P10/P50/P90.
    targets : np.ndarray
        Actual observed values. Shape should broadcast with q_predictions
        when the last dimension is removed, i.e., shape (...,) matching
        all dimensions of q_predictions except the last.
    quantiles : list[float]
        Quantile levels corresponding to the last axis of q_predictions.
        For example, [0.1, 0.5, 0.9] for P10/P50/P90.

    Returns
    -------
    float
        The mean CRPS value averaged over all samples and time steps.
    """
    # Convert inputs to numpy arrays
    q_predictions = np.asarray(q_predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # K = number of quantile levels used in the approximation
    num_quantiles = len(quantiles)

    # Accumulate pinball loss across all K quantile levels
    # CRPS ≈ (2/K) * Σ_{k=1}^{K} mean(pinball_loss_k)
    total_pinball = 0.0

    for i, tau in enumerate(quantiles):
        # Extract predictions for quantile level τ_k (last axis index i)
        q_hat = q_predictions[..., i]

        # Compute the residual: actual minus predicted (y - q_hat)
        residual = targets - q_hat

        # Pinball (quantile) loss for level τ:
        #   L(q, y, τ) = τ * max(y - q, 0) + (1 - τ) * max(q - y, 0)
        # Penalizes under-prediction (y > q) by factor τ,
        # and over-prediction (q > y) by factor (1 - τ)
        pinball = np.where(
            residual >= 0,
            tau * residual,          # under-prediction penalty
            (tau - 1.0) * residual,  # over-prediction penalty (equivalent to (1-τ)*(q-y))
        )

        # Add mean pinball loss for this quantile level to the running total
        total_pinball += np.mean(pinball)

    # Final CRPS = (2/K) * sum of mean pinball losses across all quantile levels
    # The factor of 2/K converts the average pinball loss to a CRPS estimate
    crps = (2.0 / num_quantiles) * total_pinball

    return float(crps)
