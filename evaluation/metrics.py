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

    MAE measures the average magnitude of forecast errors without considering
    their direction. Lower values indicate better forecast accuracy.

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
        The mean absolute error averaged over all elements.
    """
    # Convert inputs to numpy arrays for safety (handles lists, tensors, etc.)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Compute element-wise absolute differences and take the mean
    absolute_errors = np.abs(predictions - targets)
    return float(np.mean(absolute_errors))


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Squared Error between predictions and targets.

    MSE penalizes larger errors more heavily than MAE due to squaring.
    It is always non-negative and equals zero only when predictions
    perfectly match targets.

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
        The mean squared error averaged over all elements.
    """
    # Convert inputs to numpy arrays for consistent computation
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Square the differences and average across all forecast positions
    squared_errors = (predictions - targets) ** 2
    return float(np.mean(squared_errors))


def mase(
    predictions: np.ndarray,
    targets: np.ndarray,
    seasonal_period: int = 24,
) -> float:
    """Compute Mean Absolute Scaled Error using a seasonal naive baseline.

    MASE scales the MAE by the in-sample MAE of a seasonal naive forecast.
    A MASE < 1 means the model outperforms the seasonal naive baseline;
    MASE > 1 means it performs worse. The seasonal period defaults to 24
    (hourly data with daily seasonality, matching ETTh1).

    The scaling denominator is:
        mean(|y_t - y_{t - seasonal_period}|) for t = seasonal_period, ..., T-1
    computed over the target array itself (the test portion).

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
        The MASE value. Returns inf if the naive baseline error is zero
        (constant series with exact seasonal repetition).
    """
    # Flatten arrays so we can compute the seasonal naive error sequentially
    predictions = np.asarray(predictions, dtype=np.float64).flatten()
    targets = np.asarray(targets, dtype=np.float64).flatten()

    # Compute the numerator: MAE of the model's predictions
    numerator = np.mean(np.abs(predictions - targets))

    # Compute the denominator: MAE of the seasonal naive forecast on targets
    # The seasonal naive forecast at time t is simply the value at t - seasonal_period
    naive_errors = np.abs(targets[seasonal_period:] - targets[:-seasonal_period])

    # Guard against division by zero (happens if the series is perfectly periodic)
    denominator = np.mean(naive_errors)
    if denominator == 0.0:
        return float("inf")

    # MASE is the ratio of model error to naive baseline error
    return float(numerator / denominator)


def crps_quantile(
    q_predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float],
) -> float:
    """Compute CRPS approximated from quantile forecasts.

    The Continuous Ranked Probability Score is a proper scoring rule for
    probabilistic forecasts. When only discrete quantiles are available
    (e.g., P10, P50, P90), CRPS can be approximated using the quantile
    score (pinball loss) averaged across all quantile levels:

        CRPS ≈ (2 / K) * Σ_k pinball_loss(q_k, y, tau_k)

    where K is the number of quantiles, q_k is the predicted quantile value,
    y is the actual value, and tau_k is the quantile level.

    The pinball (quantile) loss for a single quantile level tau is:
        L(q, y, tau) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)

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

    # Number of quantile levels used in the approximation
    num_quantiles = len(quantiles)

    # Accumulate pinball loss across all quantile levels
    total_pinball = 0.0

    for i, tau in enumerate(quantiles):
        # Extract predictions for this quantile level (last axis index i)
        q_hat = q_predictions[..., i]

        # Compute the residual: actual minus predicted
        residual = targets - q_hat

        # Pinball loss: tau * max(residual, 0) + (1 - tau) * max(-residual, 0)
        # This penalizes under-prediction by tau and over-prediction by (1 - tau)
        pinball = np.where(
            residual >= 0,
            tau * residual,
            (tau - 1.0) * residual,
        )

        # Sum the mean pinball loss for this quantile level
        total_pinball += np.mean(pinball)

    # Average across quantile levels and scale by 2 for CRPS approximation
    # The factor of 2 converts the average pinball loss to a CRPS estimate
    crps = (2.0 / num_quantiles) * total_pinball

    return float(crps)
