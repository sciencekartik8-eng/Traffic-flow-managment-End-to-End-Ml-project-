"""
Model architecture packages for benchmark progression:
Stage 1: Baselines (Naive, Seasonal Naive, Moving Average, Ridge)
Stage 2: Tabular ML (Random Forest, XGBoost)
Stage 3: Statistical & Sequential Time-Series (SARIMA, Prophet, LSTM, GRU)
Stage 4: Spatiotemporal Modeling (STGCN, Graph WaveNet)
"""

from .baseline import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    RidgeBaselineForecaster,
)

__all__ = [
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "RidgeBaselineForecaster",
]

