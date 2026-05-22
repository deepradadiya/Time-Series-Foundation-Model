"""Evaluation package for metrics, baselines, and visualization.

This package computes forecast quality metrics (MAE, MSE, MASE, CRPS), runs classical
baselines (ARIMA, Prophet), orchestrates the full evaluation pipeline, and generates
comparison plots with prediction intervals. It consumes forecasts from the forecasting
package and produces the final results table and visualizations.
"""
