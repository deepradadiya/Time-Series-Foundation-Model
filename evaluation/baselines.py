"""Classical baseline forecasting module for comparison with the foundation model.

This module implements ARIMA and Prophet baselines that are evaluated on the same
ETTh1 test windows as the PatchTST zero-shot model. It also provides a seasonal
naive fallback used when ARIMA or Prophet fails to converge. Metrics (MAE, MSE,
MASE) are computed for each baseline to enable fair comparison.

Related modules:
    - evaluation/metrics.py provides MAE, MSE, and MASE metric functions.
    - forecasting/inference.py defines the sliding window approach and
      compute_num_windows used to generate the same test windows.
    - config.py supplies CONTEXT_LENGTH (512) and FORECAST_HORIZON (96).
    - data/preprocess.py provides normalization utilities for data preparation.
"""

import warnings

import numpy as np
import pandas as pd

from config import Config


def seasonal_naive_fallback(
    history: np.ndarray,
    horizon: int,
    period: int = 24,
) -> np.ndarray:
    """Generate a seasonal naive forecast by repeating the last seasonal cycle.

    The seasonal naive method copies values from the last full seasonal period
    of the history to produce the forecast. For each forecast step t, the
    predicted value is: history[-(period - (t % period))].

    This serves as a simple fallback when ARIMA or Prophet fails to converge,
    and also as the scaling denominator for the MASE metric.

    Parameters:
        history: A 1D numpy array of historical time series values. Must have
                 length >= period to extract at least one full seasonal cycle.
        horizon: Number of future time steps to forecast (e.g., 96).
        period: Seasonal period in time steps (default 24 for hourly data with
                daily seasonality, matching ETTh1's hourly frequency).

    Returns:
        A 1D numpy array of shape (horizon,) containing the seasonal naive
        forecast values.
    """
    # Build the forecast by cycling through the last seasonal period
    forecast = np.empty(horizon, dtype=np.float64)

    for t in range(horizon):
        # Index into history: grab the value from one period ago, cycling
        # For t=0, we get history[-period]; for t=1, history[-(period-1)]; etc.
        offset = period - (t % period)
        forecast[t] = history[-offset]

    return forecast


def _select_arima_order(
    history: np.ndarray,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
) -> tuple[int, int, int]:
    """Select the best ARIMA (p, d, q) order using AIC-based stepwise search.

    This implements a simplified auto ARIMA by testing a grid of candidate
    orders and selecting the one with the lowest AIC (Akaike Information
    Criterion). The search is constrained to max_p=5, max_d=2, max_q=5.

    To keep computation tractable, we use a stepwise approach: first determine
    the differencing order d using the ADF test, then search over (p, q) pairs.

    Parameters:
        history: A 1D numpy array of the time series to fit.
        max_p: Maximum autoregressive order (default 5).
        max_d: Maximum differencing order (default 2).
        max_q: Maximum moving average order (default 5).

    Returns:
        A tuple (p, d, q) representing the best ARIMA order found.
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA

    # Step 1: Determine differencing order d using the ADF test
    # Apply successive differencing until the series is stationary (p-value < 0.05)
    d = 0
    series = history.copy()

    for candidate_d in range(max_d + 1):
        if candidate_d > 0:
            series = np.diff(series)

        try:
            adf_result = adfuller(series, maxlag=min(len(series) // 2 - 1, 12))
            p_value = adf_result[1]

            # If p-value < 0.05, the series is stationary at this differencing level
            if p_value < 0.05:
                d = candidate_d
                break
        except Exception:
            # If ADF test fails, try next differencing level
            continue

        d = candidate_d

    # Step 2: Search over (p, q) pairs with the determined d
    # Use a stepwise approach: test small orders first, expand if needed
    best_aic = float("inf")
    best_order = (1, d, 1)

    # Candidate orders to try (stepwise: start small, expand)
    candidate_orders = [
        (0, d, 0), (1, d, 0), (0, d, 1), (1, d, 1),
        (2, d, 1), (1, d, 2), (2, d, 2),
        (3, d, 1), (1, d, 3), (3, d, 2), (2, d, 3),
        (4, d, 2), (2, d, 4), (5, d, 2), (2, d, 5),
    ]

    # Filter candidates within bounds
    candidate_orders = [
        (p, dd, q) for p, dd, q in candidate_orders
        if p <= max_p and dd <= max_d and q <= max_q
    ]

    # Use only the last portion of history for faster fitting
    # ARIMA on very long series is slow; use the most recent data
    fit_data = history[-min(len(history), 2000):]

    for order in candidate_orders:
        try:
            model = ARIMA(fit_data, order=order)
            result = model.fit()

            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            # Skip orders that fail to fit
            continue

    return best_order


def run_arima_baseline(
    train: np.ndarray,
    test_windows: list[tuple[np.ndarray, np.ndarray]],
    horizon: int = Config.FORECAST_HORIZON,
) -> np.ndarray:
    """Run auto ARIMA baseline on each test window and return point forecasts.

    For each test window, ARIMA is fit on the training data plus the context
    portion of that window, then generates a point forecast for the next
    `horizon` steps. If ARIMA fails to converge, the seasonal naive fallback
    (period=24) is used instead.

    Auto ARIMA searches for the best (p, d, q) order with constraints:
    max p=5, max d=2, max q=5. Order selection uses AIC-based stepwise search.

    Parameters:
        train: A 1D numpy array of training data (used as base history).
        test_windows: A list of tuples (context, target) where:
            - context: 1D array of shape (context_length,) — input history
            - target: 1D array of shape (horizon,) — ground truth future values
        horizon: Number of future steps to forecast (default 96).

    Returns:
        A 2D numpy array of shape (num_windows, horizon) containing point
        forecasts for each test window.
    """
    # Import statsmodels ARIMA here to avoid import errors if not installed
    from statsmodels.tsa.arima.model import ARIMA

    # Store forecasts for all windows
    num_windows = len(test_windows)
    all_forecasts = np.empty((num_windows, horizon), dtype=np.float64)

    for window_idx, (context, _target) in enumerate(test_windows):
        # Combine training data with the context portion for this window
        # This gives ARIMA the full history up to the forecast origin
        history = np.concatenate([train, context])

        try:
            # Suppress convergence warnings from statsmodels during fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Select the best ARIMA order using AIC-based stepwise search
                # Constrained to max_p=5, max_d=2, max_q=5
                best_order = _select_arima_order(
                    history=history, max_p=5, max_d=2, max_q=5
                )

                # Fit ARIMA with the selected order on recent history
                # Use the most recent portion to keep fitting tractable
                fit_data = history[-min(len(history), 2000):]
                model = ARIMA(fit_data, order=best_order)
                result = model.fit()

                # Generate point forecast for the next `horizon` steps
                forecast = result.forecast(steps=horizon)
                all_forecasts[window_idx] = forecast

        except Exception as e:
            # ARIMA failed to converge — use seasonal naive fallback
            print(
                f"[ARIMA] Window {window_idx}: fitting failed ({type(e).__name__}: {e}). "
                f"Using seasonal naive fallback (period=24)."
            )
            all_forecasts[window_idx] = seasonal_naive_fallback(
                history=history, horizon=horizon, period=24
            )

    return all_forecasts


def run_prophet_baseline(
    train: np.ndarray,
    test_windows: list[tuple[np.ndarray, np.ndarray]],
    horizon: int = Config.FORECAST_HORIZON,
) -> np.ndarray:
    """Run Facebook Prophet baseline on each test window and return point forecasts.

    For each test window, Prophet is fit on the training data plus the context
    portion of that window, then generates a point forecast (yhat) for the next
    `horizon` steps. If Prophet fails, the seasonal naive fallback (period=24)
    is used instead.

    Parameters:
        train: A 1D numpy array of training data (used as base history).
        test_windows: A list of tuples (context, target) where:
            - context: 1D array of shape (context_length,) — input history
            - target: 1D array of shape (horizon,) — ground truth future values
        horizon: Number of future steps to forecast (default 96).

    Returns:
        A 2D numpy array of shape (num_windows, horizon) containing point
        forecasts for each test window.
    """
    # Import Prophet here to avoid import errors if not installed
    from prophet import Prophet

    # Store forecasts for all windows
    num_windows = len(test_windows)
    all_forecasts = np.empty((num_windows, horizon), dtype=np.float64)

    for window_idx, (context, _target) in enumerate(test_windows):
        # Combine training data with the context portion for this window
        history = np.concatenate([train, context])

        try:
            # Suppress Prophet's verbose logging output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Prophet requires a DataFrame with 'ds' (datetime) and 'y' (value) columns
                # We create synthetic hourly timestamps since we only need the forecast values
                dates = pd.date_range(
                    start="2000-01-01", periods=len(history), freq="h"
                )
                df_train = pd.DataFrame({"ds": dates, "y": history})

                # Initialize Prophet with suppressed logging
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                )
                model.fit(df_train)

                # Create future dataframe for the forecast horizon
                future_dates = pd.date_range(
                    start=dates[-1] + pd.Timedelta(hours=1),
                    periods=horizon,
                    freq="h",
                )
                df_future = pd.DataFrame({"ds": future_dates})

                # Generate point forecast (yhat column)
                forecast_df = model.predict(df_future)
                forecast = forecast_df["yhat"].values
                all_forecasts[window_idx] = forecast

        except Exception as e:
            # Prophet failed — use seasonal naive fallback
            print(
                f"[Prophet] Window {window_idx}: fitting failed ({type(e).__name__}: {e}). "
                f"Using seasonal naive fallback (period=24)."
            )
            all_forecasts[window_idx] = seasonal_naive_fallback(
                history=history, horizon=horizon, period=24
            )

    return all_forecasts


def compute_baseline_metrics(
    forecasts: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    seasonal_period: int = 24,
) -> dict[str, float]:
    """Compute MAE, MSE, and MASE for a set of baseline forecasts.

    Metrics are averaged across all test windows and all forecast steps.
    MASE is scaled by the mean absolute error of a seasonal naive forecast
    with the specified seasonal period.

    Parameters:
        forecasts: A 2D numpy array of shape (num_windows, horizon) with
                   point forecasts from the baseline method.
        targets: A 2D numpy array of shape (num_windows, horizon) with
                 ground truth values for each test window.
        train: A 1D numpy array of training data used to compute the MASE
               scaling factor (seasonal naive error on training set).
        seasonal_period: Period for the seasonal naive reference in MASE
                         calculation (default 24 for hourly/daily).

    Returns:
        A dictionary with keys "mae", "mse", and "mase" containing the
        computed metric values as floats.
    """
    # Compute Mean Absolute Error: average |forecast - actual| across all steps
    mae_value = float(np.mean(np.abs(forecasts - targets)))

    # Compute Mean Squared Error: average (forecast - actual)^2 across all steps
    mse_value = float(np.mean((forecasts - targets) ** 2))

    # Compute MASE: MAE scaled by the in-sample seasonal naive error
    # The denominator is the mean absolute error of a one-step seasonal naive
    # forecast on the training data: mean(|y_t - y_{t-period}|)
    naive_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])
    naive_mae = np.mean(naive_errors)

    # Avoid division by zero if the training data is constant
    if naive_mae == 0.0:
        mase_value = float("inf")
    else:
        mase_value = mae_value / naive_mae

    return {"mae": mae_value, "mse": mse_value, "mase": mase_value}


def extract_test_windows(
    data: np.ndarray,
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract test windows from a data array using the sliding window approach.

    This produces the same windows used by the zero-shot evaluator, ensuring
    fair comparison between the foundation model and classical baselines.

    Parameters:
        data: A 1D numpy array of the full test split (or combined val+test).
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps per window (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        A list of (context, target) tuples where:
            - context: 1D array of shape (context_length,)
            - target: 1D array of shape (forecast_horizon,)
    """
    windows: list[tuple[np.ndarray, np.ndarray]] = []

    # Minimum data length needed for at least one window
    min_required = context_length + forecast_horizon

    if len(data) < min_required:
        return windows

    # Number of valid windows: floor((T - context - horizon) / stride) + 1
    num_windows = (len(data) - context_length - forecast_horizon) // stride + 1

    for i in range(num_windows):
        # Start of the context for this window
        start = i * stride

        # Extract context (input) and target (ground truth future)
        context = data[start : start + context_length]
        target = data[start + context_length : start + context_length + forecast_horizon]

        windows.append((context, target))

    return windows


def run_all_baselines(
    train: np.ndarray,
    test_data: np.ndarray,
    context_length: int = Config.CONTEXT_LENGTH,
    forecast_horizon: int = Config.FORECAST_HORIZON,
    stride: int = Config.FORECAST_HORIZON,
) -> dict[str, dict[str, float]]:
    """Run all classical baselines and compute metrics on the same test windows.

    This is the main entry point for baseline evaluation. It extracts test
    windows from the test data, runs ARIMA and Prophet on each window, and
    computes MAE, MSE, and MASE for each method.

    Parameters:
        train: A 1D numpy array of training data for fitting baselines.
        test_data: A 1D numpy array of the test split for evaluation.
        context_length: Number of input time steps per window (default 512).
        forecast_horizon: Number of future steps to forecast (default 96).
        stride: Step size between consecutive windows (default 96).

    Returns:
        A dictionary mapping baseline names to their metric dictionaries:
        {
            "arima": {"mae": float, "mse": float, "mase": float},
            "prophet": {"mae": float, "mse": float, "mase": float},
        }
    """
    # Extract test windows (same windows used by zero-shot evaluator)
    test_windows = extract_test_windows(
        data=test_data,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )

    if len(test_windows) == 0:
        print("[Baselines] No valid test windows found. Test data may be too short.")
        return {"arima": {"mae": 0.0, "mse": 0.0, "mase": 0.0},
                "prophet": {"mae": 0.0, "mse": 0.0, "mase": 0.0}}

    # Collect ground truth targets into a 2D array for metric computation
    targets = np.array([target for _, target in test_windows])

    # -------------------------------------------------------------------------
    # Run ARIMA baseline
    # -------------------------------------------------------------------------
    print(f"[Baselines] Running ARIMA on {len(test_windows)} test windows...")
    arima_forecasts = run_arima_baseline(
        train=train, test_windows=test_windows, horizon=forecast_horizon
    )
    arima_metrics = compute_baseline_metrics(
        forecasts=arima_forecasts,
        targets=targets,
        train=train,
        seasonal_period=24,
    )
    print(f"[Baselines] ARIMA — MAE: {arima_metrics['mae']:.4f}, "
          f"MSE: {arima_metrics['mse']:.4f}, MASE: {arima_metrics['mase']:.4f}")

    # -------------------------------------------------------------------------
    # Run Prophet baseline
    # -------------------------------------------------------------------------
    print(f"[Baselines] Running Prophet on {len(test_windows)} test windows...")
    prophet_forecasts = run_prophet_baseline(
        train=train, test_windows=test_windows, horizon=forecast_horizon
    )
    prophet_metrics = compute_baseline_metrics(
        forecasts=prophet_forecasts,
        targets=targets,
        train=train,
        seasonal_period=24,
    )
    print(f"[Baselines] Prophet — MAE: {prophet_metrics['mae']:.4f}, "
          f"MSE: {prophet_metrics['mse']:.4f}, MASE: {prophet_metrics['mase']:.4f}")

    return {
        "arima": arima_metrics,
        "prophet": prophet_metrics,
    }
