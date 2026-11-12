"""
Stage 1: Baseline benchmark forecasting models.

Provides canonical reference baselines:
- Naive Forecaster: Forecast equals last observed step y_{t}
- Seasonal Naive Forecaster: Forecast equals observation at same time yesterday y_{t - 96}
- Moving Average Forecaster: Forecast equals rolling average of recent k steps
- Ridge Linear Forecaster: Regularized linear regression on autoregressive lags
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class NaiveForecaster:
    """Predicts future steps using the most recent observation y_t (Persistence Model)."""

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        return self

    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Args:
            y_history: Array of historical observations.
            horizon: Steps ahead to project.
        """
        last_val = y_history[-1]
        return np.full(shape=(horizon,), fill_value=last_val)


class SeasonalNaiveForecaster:
    """Predicts future steps using identical steps from the previous seasonal cycle (e.g., 96 steps for 24h)."""

    def __init__(self, season_length: int = 96) -> None:
        self.season_length = season_length

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        return self

    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        if len(y_history) < self.season_length:
            raise ValueError(f"History length ({len(y_history)}) must be >= season_length ({self.season_length}).")
        # Extract corresponding seasonal points from previous day
        start_idx = -self.season_length
        end_idx = start_idx + horizon
        if end_idx == 0:
            return y_history[start_idx:]
        elif end_idx < 0:
            return y_history[start_idx:end_idx]
        else:
            # Multi-day wrapping if horizon exceeds season length
            return np.array([y_history[-self.season_length + (i % self.season_length)] for i in range(horizon)])


class MovingAverageForecaster:
    """Forecasts future intervals as the arithmetic mean of the preceding k steps."""

    def __init__(self, window_size: int = 4) -> None:
        self.window_size = window_size

    def fit(self, y: np.ndarray) -> "MovingAverageForecaster":
        return self

    def predict(self, y_history: np.ndarray, horizon: int = 1) -> np.ndarray:
        if len(y_history) < self.window_size:
            avg_val = np.mean(y_history)
        else:
            avg_val = np.mean(y_history[-self.window_size:])
        return np.full(shape=(horizon,), fill_value=avg_val)


class RidgeBaselineForecaster:
    """Regularized linear regression model mapping autoregressive lags to target."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaselineForecaster":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

