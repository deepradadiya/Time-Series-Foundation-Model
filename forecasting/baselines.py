"""Classical baseline forecasting runners for the evaluation pipeline.

This module provides Naive, ARIMA, and Prophet baseline implementations with
timing instrumentation and metric computation. It wraps the existing logic in
evaluation/baselines.py and adds the Naive baseline for a complete set of
classical comparisons against the PatchTST foundation model.

All baselines use consistent windowing parameters:
    - context_length=512 (input window size)
    - forecast_horizon=96 (prediction length)
    - stride=96 (non-overlapping windows)

Related modules:
    - evaluation/baselines.py provides the core ARIMA, Prophet, and seasonal
      naive implementations that this module wraps.
    - evaluation/metrics.py provides MAE, MSE, and MASE metric functions.
    - config.py supplies CONTEXT_LENGTH (512) and FORECAST_HORIZON (96).
"""

import time
import warnings
from typing import Any

import numpy as np

from config import Config
from evaluation.baselines import (
    extract_test_windows,
    seasonal_naive_fallback,
)
from evaluation.metrics import mae, mse, mase


def naive_forecast(context: np.ndarray, horizon: int = 96) -> np.ndarray:
    """Predict the last value of context for all horizon steps.

    The Naive baseline repeats the final observed value for every future time
    step. This serves as the minimum performance floor — any useful model
    should outperform this trivial prediction.

    Parameters:
        context: 1D array of historical values (length >= 1).
        horizon: Number of future steps to predict (default 96).

    Returns:
        1D array of shape (horizon,) filled with context[-1].

    Raises:
        ValueError: If context is empty (length 0).
    """
    context = np.asarray(context, dtype=np.float64)

    if len(context) == 0:
        raise ValueError(
            "Context window must contain at least one value. "
            "Received an empty context array (length 0)."
        )

    return np.full(horizon, context[-1], dtype=np.float64)


def run_naive_baseline(
    test_data: np.ndarray,
    context_length: int = 512,
    forecast_horizon: int = 96,
    stride: int = 96,
) -> dict[str, Any]:
    """Run Naive baseline on all test windows, return forecasts + metrics + timing.

    For each sliding window in the test data, the Naive baseline predicts the
    last value of the context window repeated for the entire forecast horizon.
    Metrics (MAE, MSE, MASE) are computed across all windows.

    Parameters:
        test_data: 1D numpy array of the test split.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to forecast (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        Dictionary with keys:
            - "forecasts": 2D numpy array of shape (num_windows, forecast_horizon)
            - "metrics": dict with "mae", "mse", "mase" float values
            - "inference_time": float, wall-clock seconds for all forecasts
    """
    test_data = np.asarray(test_data, dtype=np.float64)

    # Extract test windows using the same approach as other baselines
    test_windows = extract_test_windows(
        data=test_data,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if len(test_windows) == 0:
        return {
            "forecasts": np.empty((0, forecast_horizon), dtype=np.float64),
            "metrics": {"mae": 0.0, "mse": 0.0, "mase": 0.0},
            "inference_time": 0.0,
        }

    num_windows = len(test_windows)
    all_forecasts = np.empty((num_windows, forecast_horizon), dtype=np.float64)
    targets = np.empty((num_windows, forecast_horizon), dtype=np.float64)

    # Time the inference (generating all forecasts)
    start_time = time.time()

    for window_idx, (context, target) in enumerate(test_windows):
        all_forecasts[window_idx] = naive_forecast(context, horizon=forecast_horizon)
        targets[window_idx] = target

    inference_time = time.time() - start_time

    # Compute metrics averaged across all windows and forecast steps
    mae_value = mae(all_forecasts, targets)
    mse_value = mse(all_forecasts, targets)
    mase_value = mase(all_forecasts, targets, seasonal_period=24)

    return {
        "forecasts": all_forecasts,
        "metrics": {"mae": mae_value, "mse": mse_value, "mase": mase_value},
        "inference_time": round(inference_time, 2),
    }


def run_arima_baseline_eval(
    train: np.ndarray,
    test_data: np.ndarray,
    context_length: int = 512,
    forecast_horizon: int = 96,
    stride: int = 96,
) -> dict[str, Any]:
    """Run ARIMA (pmdarima auto_arima) on all test windows.

    For each test window, ARIMA is fit using pmdarima's auto_arima with
    search bounds max_p=5, max_d=2, max_q=5. If auto_arima fails for any
    window, the seasonal naive fallback (period=24) is used and a warning
    is printed.

    This wraps the existing evaluation/baselines.py ARIMA logic but uses
    pmdarima's auto_arima for more robust parameter selection.

    Parameters:
        train: 1D numpy array of training data (used as base history).
        test_data: 1D numpy array of the test split for evaluation.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to forecast (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        Dictionary with keys:
            - "forecasts": 2D numpy array of shape (num_windows, forecast_horizon)
            - "metrics": dict with "mae", "mse", "mase" float values
            - "inference_time": float, wall-clock seconds for fitting + forecasting
    """
    train = np.asarray(train, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)

    # Extract test windows
    test_windows = extract_test_windows(
        data=test_data,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if len(test_windows) == 0:
        return {
            "forecasts": np.empty((0, forecast_horizon), dtype=np.float64),
            "metrics": {"mae": 0.0, "mse": 0.0, "mase": 0.0},
            "inference_time": 0.0,
        }

    num_windows = len(test_windows)
    all_forecasts = np.empty((num_windows, forecast_horizon), dtype=np.float64)
    targets = np.empty((num_windows, forecast_horizon), dtype=np.float64)

    # Time the full ARIMA evaluation (fitting + forecasting across all windows)
    start_time = time.time()

    for window_idx, (context, target) in enumerate(test_windows):
        targets[window_idx] = target

        # Combine training data with the context portion for this window
        history = np.concatenate([train, context])

        try:
            # Import pmdarima for auto_arima
            import pmdarima as pm

            # Suppress convergence warnings during fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Use pmdarima's auto_arima with specified search bounds
                model = pm.auto_arima(
                    history[-min(len(history), 2000):],  # Use recent history
                    max_p=5,
                    max_d=2,
                    max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                )

                # Generate point forecast for the next forecast_horizon steps
                forecast = model.predict(n_periods=forecast_horizon)
                all_forecasts[window_idx] = forecast

        except Exception as e:
            # ARIMA failed — use seasonal naive fallback (period=24)
            print(
                f"[ARIMA] Window {window_idx}: fitting failed "
                f"({type(e).__name__}). Using seasonal naive fallback (period=24)."
            )
            all_forecasts[window_idx] = seasonal_naive_fallback(
                history=history, horizon=forecast_horizon, period=24
            )

    inference_time = time.time() - start_time

    # Compute metrics averaged across all windows
    mae_value = mae(all_forecasts, targets)
    mse_value = mse(all_forecasts, targets)
    mase_value = mase(all_forecasts, targets, seasonal_period=24)

    print(
        f"[ARIMA] Evaluation complete — MAE: {mae_value:.4f}, "
        f"MSE: {mse_value:.4f}, MASE: {mase_value:.4f}, "
        f"Time: {inference_time:.2f}s"
    )

    return {
        "forecasts": all_forecasts,
        "metrics": {"mae": mae_value, "mse": mse_value, "mase": mase_value},
        "inference_time": round(inference_time, 2),
    }


def run_prophet_baseline_eval(
    train: np.ndarray,
    test_data: np.ndarray,
    context_length: int = 512,
    forecast_horizon: int = 96,
    stride: int = 96,
) -> dict[str, Any]:
    """Run Prophet on all test windows.

    For each test window, Prophet is fit on the training data concatenated
    with the context portion, using daily_seasonality=True,
    weekly_seasonality=True, yearly_seasonality=False. If Prophet fails for
    any window, the seasonal naive fallback (period=24) is used and a warning
    is printed.

    This wraps the existing evaluation/baselines.py Prophet logic with the
    specified seasonality configuration.

    Parameters:
        train: 1D numpy array of training data (used as base history).
        test_data: 1D numpy array of the test split for evaluation.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to forecast (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        Dictionary with keys:
            - "forecasts": 2D numpy array of shape (num_windows, forecast_horizon)
            - "metrics": dict with "mae", "mse", "mase" float values
            - "inference_time": float, wall-clock seconds for fitting + forecasting
    """
    train = np.asarray(train, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)

    # Extract test windows
    test_windows = extract_test_windows(
        data=test_data,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if len(test_windows) == 0:
        return {
            "forecasts": np.empty((0, forecast_horizon), dtype=np.float64),
            "metrics": {"mae": 0.0, "mse": 0.0, "mase": 0.0},
            "inference_time": 0.0,
        }

    num_windows = len(test_windows)
    all_forecasts = np.empty((num_windows, forecast_horizon), dtype=np.float64)
    targets = np.empty((num_windows, forecast_horizon), dtype=np.float64)

    # Time the full Prophet evaluation (fitting + forecasting across all windows)
    start_time = time.time()

    for window_idx, (context, target) in enumerate(test_windows):
        targets[window_idx] = target

        # Combine training data with the context portion for this window
        history = np.concatenate([train, context])

        try:
            import pandas as pd
            from prophet import Prophet

            # Suppress Prophet's verbose logging output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Prophet requires a DataFrame with 'ds' (datetime) and 'y' (value)
                # Create synthetic hourly timestamps
                dates = pd.date_range(
                    start="2000-01-01", periods=len(history), freq="h"
                )
                df_train = pd.DataFrame({"ds": dates, "y": history})

                # Initialize Prophet with specified seasonality configuration
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                )
                model.fit(df_train)

                # Create future dataframe for the forecast horizon
                future_dates = pd.date_range(
                    start=dates[-1] + pd.Timedelta(hours=1),
                    periods=forecast_horizon,
                    freq="h",
                )
                df_future = pd.DataFrame({"ds": future_dates})

                # Generate point forecast (yhat column)
                forecast_df = model.predict(df_future)
                forecast = forecast_df["yhat"].values
                all_forecasts[window_idx] = forecast

        except Exception as e:
            # Prophet failed — use seasonal naive fallback (period=24)
            print(
                f"[Prophet] Window {window_idx}: fitting failed "
                f"({type(e).__name__}). Using seasonal naive fallback (period=24)."
            )
            all_forecasts[window_idx] = seasonal_naive_fallback(
                history=history, horizon=forecast_horizon, period=24
            )

    inference_time = time.time() - start_time

    # Compute metrics averaged across all windows
    mae_value = mae(all_forecasts, targets)
    mse_value = mse(all_forecasts, targets)
    mase_value = mase(all_forecasts, targets, seasonal_period=24)

    print(
        f"[Prophet] Evaluation complete — MAE: {mae_value:.4f}, "
        f"MSE: {mse_value:.4f}, MASE: {mase_value:.4f}, "
        f"Time: {inference_time:.2f}s"
    )

    return {
        "forecasts": all_forecasts,
        "metrics": {"mae": mae_value, "mse": mse_value, "mase": mase_value},
        "inference_time": round(inference_time, 2),
    }
