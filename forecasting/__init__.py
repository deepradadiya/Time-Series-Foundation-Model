"""Forecasting package for probabilistic predictions.

This package implements the probabilistic forecast head (P10/P50/P90 quantile
regression), fine-tuning on ETTh1, and zero-shot inference. It takes the pretrained
model from the model package and produces quantile forecasts evaluated by the
evaluation package.
"""
