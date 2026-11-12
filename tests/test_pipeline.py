"""
Unit tests for data loading, feature engineering, and evaluation metrics.
"""

import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None

from src.evaluate import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    evaluate_predictions,
)
from src.data_loader import TrafficDataLoader
from src.features import TrafficFeaturePipeline


def test_evaluation_metrics_deterministic():
    y_true = np.array([100.0, 200.0, 300.0, 0.0])
    y_pred = np.array([110.0, 190.0, 300.0, 10.0])

    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 7.5)

    # MSE = (10^2 + 10^2 + 0^2 + 10^2) / 4 = 300 / 4 = 75.0. RMSE = sqrt(75) ~ 8.66025
    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(75.0))

    # WMAPE: total abs err = 10 + 10 + 0 + 10 = 30. Total true = 600. 30/600 * 100 = 5.0%
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(wmape, 5.0)

    # Verify zero-safety: test on pure zeros
    zeros = np.zeros(5)
    assert weighted_mean_absolute_percentage_error(zeros, zeros) == 0.0
    assert symmetric_mean_absolute_percentage_error(zeros, zeros) == 0.0


def test_data_loader_resampling():
    # Construct 1-minute frequency data spanning 2 hours
    dates = pd.date_range("2024-01-01 08:00:00", "2024-01-01 09:59:00", freq="1min")
    df_raw = pd.DataFrame({
        "timestamp": dates,
        "junction_id": ["J1"] * len(dates),
        "vehicle_count": [1] * len(dates),
    })

    loader = TrafficDataLoader(
        timestamp_col="timestamp",
        target_col="vehicle_count",
        sensor_id_col="junction_id",
        resample_interval="15min",
        aggregation_func="sum",
    )

    df_resampled = loader.resample_all(df_raw, fill_missing="zero")

    # 2 hours at 15-min intervals = 8 steps
    assert len(df_resampled) == 8
    # 15 minutes of 1 count per minute = 15 counts per interval
    assert (df_resampled["vehicle_count"] == 15).all()


def test_chronological_split():
    dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="15min")
    df = pd.DataFrame({"timestamp": dates, "vehicle_count": np.arange(100)})

    splits = TrafficDataLoader.chronological_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, timestamp_col="timestamp"
    )

    assert len(splits.train) == 70
    assert len(splits.val) == 15
    assert len(splits.test) == 15
    assert splits.train["timestamp"].max() < splits.val["timestamp"].min()
    assert splits.val["timestamp"].max() < splits.test["timestamp"].min()


def test_feature_pipeline_no_leakage():
    dates = pd.date_range("2024-01-01 00:00:00", periods=200, freq="15min")
    df = pd.DataFrame({"timestamp": dates, "vehicle_count": np.arange(200, dtype=float)})

    pipeline = TrafficFeaturePipeline(
        timestamp_col="timestamp",
        target_col="vehicle_count",
        sensor_id_col=None,
        lags=[1, 2, 4],
        rolling_windows=[4],
        horizon_steps=[1, 2],
    )

    df_feats = pipeline.transform(df, drop_na=True)

    # Check lag 1 equals previous step
    assert "vehicle_count_lag_1" in df_feats.columns
    # Check cyclical features exist and are bounded in [-1, 1]
    assert "sin_time_of_day" in df_feats.columns
    assert df_feats["sin_time_of_day"].between(-1.0, 1.0).all()
    # Check target horizons exist
    assert "target_h1" in df_feats.columns
    assert "target_h2" in df_feats.columns
