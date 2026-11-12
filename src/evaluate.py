"""
Evaluation metrics suite for urban traffic flow forecasting.

Provides robust, zero-safe performance indicators:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Weighted Mean Absolute Percentage Error (WMAPE - Industry Gold Standard for Traffic)
- Symmetric Mean Absolute Percentage Error (sMAPE)
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE).

    MAE = (1 / n) * sum(|y_true - y_pred|)
    """
    y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE).

    RMSE = sqrt((1 / n) * sum((y_true - y_pred)^2))
    """
    y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def weighted_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Computes Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE = sum(|y_true - y_pred|) / (sum(y_true) + eps) * 100

    Unlike standard MAPE which divides by individual zero or near-zero observations
    (causing arithmetic explosion in nocturnal traffic intervals), WMAPE weights errors
    by overall aggregate volume, making it the industry standard in traffic engineering.

    Args:
        y_true: Ground truth target array.
        y_pred: Model predictions array.
        eps: Small epsilon preventing division by zero if entire test sequence is 0.

    Returns:
        float: WMAPE expressed as a percentage [0, 100+].
    """
    y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total_abs_error = np.sum(np.abs(y_t - y_p))
    total_volume = np.sum(np.abs(y_t))
    if total_volume < eps:
        return 0.0 if total_abs_error < eps else 100.0
    return float((total_abs_error / total_volume) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Computes Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE = (100 / n) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )

    Bound within [0, 200%], handling scale symmetry between over- and under-predictions.

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        eps: Epsilon avoiding zero division.

    Returns:
        float: sMAPE value as a percentage.
    """
    y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_t) + np.abs(y_p) + eps
    numerator = 2.0 * np.abs(y_t - y_p)
    return float(np.mean(numerator / denominator) * 100.0)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
) -> Dict[str, float]:
    """Evaluates forecasts across all project metrics.

    Args:
        y_true: Ground truth actual values.
        y_pred: Predicted forecast values.

    Returns:
        Dict[str, float]: Dictionary with MAE, RMSE, WMAPE, and sMAPE.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    return {
        "mae": round(mean_absolute_error(y_t, y_p), 4),
        "rmse": round(root_mean_squared_error(y_t, y_p), 4),
        "wmape": round(weighted_mean_absolute_percentage_error(y_t, y_p), 4),
        "smape": round(symmetric_mean_absolute_percentage_error(y_t, y_p), 4),
    }


def evaluate_multistep_forecast(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    horizon_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluates multi-step forecasting horizons (e.g., 15m, 30m, 45m, 60m).

    Args:
        y_true_matrix: 2D array of shape (N, num_horizons).
        y_pred_matrix: 2D array of shape (N, num_horizons).
        horizon_labels: Custom names for horizons, e.g. ['t+1 (15m)', 't+2 (30m)', ...].

    Returns:
        pd.DataFrame: Metric summary table indexed by forecast horizon.
    """
    y_true_m = np.asarray(y_true_matrix)
    y_pred_m = np.asarray(y_pred_matrix)

    num_horizons = y_true_m.shape[1]
    if horizon_labels is None:
        horizon_labels = [f"Horizon t+{i+1} ({(i+1)*15}min)" for i in range(num_horizons)]

    results: List[Dict[str, Union[str, float]]] = []
    for i in range(num_horizons):
        metrics = evaluate_predictions(y_true_m[:, i], y_pred_m[:, i])
        metrics["horizon"] = horizon_labels[i]
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index("horizon")
    # Add overall average across horizons
    mean_row = df_results.mean().to_frame().T
    mean_row.index = ["Average"]
    return pd.concat([df_results, mean_row])

