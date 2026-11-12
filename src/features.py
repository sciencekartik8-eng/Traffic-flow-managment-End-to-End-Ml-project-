"""
Feature engineering pipeline for urban traffic flow forecasting.

Implements:
1. Cyclical continuous time encoding (Sine/Cosine transformations)
2. Autoregressive multi-horizon lag creation (t-1 to t-4, t-96 for 24h daily periodicity)
3. Zero-leakage shifted rolling window statistics (mean, std, min, max)
4. Exogenous contextual features (Indian national holidays, peak-hour masks, weekend flags)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("feature_pipeline")


class TrafficFeaturePipeline:
    """Feature engineering pipeline designed strictly to prevent temporal lookahead bias.

    Attributes:
        timestamp_col: Name of datetime column.
        target_col: Name of traffic volume/flow target column.
        sensor_id_col: Grouping column for multi-junction datasets.
        lags: List of lag step offsets (e.g., [1, 2, 3, 4, 96]).
        rolling_windows: List of window lengths for rolling aggregations.
        horizon_steps: Target forecast horizons to generate (e.g., [1, 2, 3, 4]).
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        target_col: str = "vehicle_count",
        sensor_id_col: Optional[str] = "junction_id",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        horizon_steps: Optional[List[int]] = None,
    ) -> None:
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.sensor_id_col = sensor_id_col
        self.lags = lags if lags is not None else [1, 2, 3, 4, 8, 96]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [4, 12, 96]
        self.horizon_steps = horizon_steps if horizon_steps is not None else [1, 2, 3, 4]

    def add_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes periodic temporal dimensions onto continuous 2D Cartesian circles.

        Prevents boundary discontinuity (e.g., 23:59 transitioning to 00:00, or Sunday to Monday).

        Transforms:
        - Minute of Day: Period = 1440 minutes
        - Hour of Day: Period = 24 hours
        - Day of Week: Period = 7 days
        - Month of Year: Period = 12 months

        Args:
            df: DataFrame containing the timestamp column.

        Returns:
            pd.DataFrame: Augmented with sin/cos cyclical columns.
        """
        df = df.copy()
        ts = pd.to_datetime(df[self.timestamp_col])

        # 1. Minute of Day (continuous intra-day diurnal cycle)
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        df["sin_time_of_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        df["cos_time_of_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        # 2. Day of Week (weekly seasonality: Monday=0, Sunday=6)
        day_of_week = ts.dt.dayofweek
        df["sin_day_of_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["cos_day_of_week"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # 3. Month of Year (annual seasonality)
        month = ts.dt.month
        df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        return df

    def add_autoregressive_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs historical lag features per sensor/junction.

        Lags include short-term steps (t-1, t-2, t-3, t-4) and seasonal steps
        (t-96 corresponding to 24 hours prior at 15-minute intervals).

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Augmented with lagged columns.
        """
        df = df.copy()

        def _compute_lags(group: pd.DataFrame) -> pd.DataFrame:
            for lag in self.lags:
                group[f"{self.target_col}_lag_{lag}"] = group[self.target_col].shift(lag)
            return group

        if self.sensor_id_col and self.sensor_id_col in df.columns:
            df = df.groupby(self.sensor_id_col, group_keys=False).apply(_compute_lags)
        else:
            df = _compute_lags(df)

        return df

    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Computes rolling window statistics on historical observations.

        CRITICAL FOR CAUSALITY:
        Rolling calculations are applied to the 1-step shifted series (.shift(1))
        to ensure the observation at time t is NEVER present in the summary statistics
        used to forecast horizon targets.

        Args:
            df: Input DataFrame.
            metrics: Aggregations to compute (default: ['mean', 'std', 'min', 'max']).

        Returns:
            pd.DataFrame: Augmented with rolling statistic columns.
        """
        df = df.copy()
        if metrics is None:
            metrics = ["mean", "std", "min", "max"]

        def _compute_rolling(group: pd.DataFrame) -> pd.DataFrame:
            # Shift by 1 step to exclude current observation
            shifted = group[self.target_col].shift(1)
            for window in self.rolling_windows:
                roller = shifted.rolling(window=window, min_periods=max(1, window // 2))
                if "mean" in metrics:
                    group[f"{self.target_col}_roll_mean_{window}"] = roller.mean()
                if "std" in metrics:
                    group[f"{self.target_col}_roll_std_{window}"] = roller.std().fillna(0.0)
                if "min" in metrics:
                    group[f"{self.target_col}_roll_min_{window}"] = roller.min()
                if "max" in metrics:
                    group[f"{self.target_col}_roll_max_{window}"] = roller.max()
            return group

        if self.sensor_id_col and self.sensor_id_col in df.columns:
            df = df.groupby(self.sensor_id_col, group_keys=False).apply(_compute_rolling)
        else:
            df = _compute_rolling(df)

        return df

    def add_exogenous_calendar_features(
        self,
        df: pd.DataFrame,
        morning_peak: Tuple[int, int] = (8, 11),
        evening_peak: Tuple[int, int] = (17, 21),
    ) -> pd.DataFrame:
        """Extracts urban contextual and calendar indicators relevant to Indian traffic.

        Features include:
        - Weekend flag (Saturday/Sunday)
        - Morning peak congestion window (08:00 - 11:00)
        - Evening peak congestion window (17:00 - 21:00)
        - Indian national and state public holiday flags (Republic Day, Independence Day, Diwali, etc.)

        Args:
            df: Input DataFrame.
            morning_peak: Tuple of (start_hour, end_hour) for morning commute.
            evening_peak: Tuple of (start_hour, end_hour) for evening commute.

        Returns:
            pd.DataFrame: Augmented with binary indicator columns.
        """
        df = df.copy()
        ts = pd.to_datetime(df[self.timestamp_col])

        # Weekend indicator
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

        # Commute peak hours
        hour = ts.dt.hour
        df["is_morning_peak"] = (
            (hour >= morning_peak[0]) & (hour < morning_peak[1]) & (df["is_weekend"] == 0)
        ).astype(int)
        df["is_evening_peak"] = (
            (hour >= evening_peak[0]) & (hour < evening_peak[1]) & (df["is_weekend"] == 0)
        ).astype(int)

        # Indian National / Gazetted Holidays
        try:
            import holidays
            unique_years = ts.dt.year.unique().tolist()
            india_holidays = holidays.India(years=unique_years)
            df["is_holiday"] = ts.dt.date.isin(india_holidays).astype(int)
        except ImportError:
            logger.warning("Package 'holidays' not found. Using fallback hardcoded holiday calendar.")
            # Fallback for primary Indian national holidays (Republic, Independence, Gandhi Jayanti)
            dates = ts.dt.strftime("%m-%d")
            primary_holidays = {"01-26", "08-15", "10-02"}
            df["is_holiday"] = dates.isin(primary_holidays).astype(int)

        return df

    def add_forecast_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates future multi-step targets for direct forecasting.

        For each step h in horizon_steps:
        target_h1 = flow at t+1 (15 min ahead)
        target_h2 = flow at t+2 (30 min ahead)
        target_h3 = flow at t+3 (45 min ahead)
        target_h4 = flow at t+4 (60 min ahead)

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: Augmented with target columns.
        """
        df = df.copy()

        def _shift_targets(group: pd.DataFrame) -> pd.DataFrame:
            for h in self.horizon_steps:
                group[f"target_h{h}"] = group[self.target_col].shift(-h)
            return group

        if self.sensor_id_col and self.sensor_id_col in df.columns:
            df = df.groupby(self.sensor_id_col, group_keys=False).apply(_shift_targets)
        else:
            df = _shift_targets(df)

        return df

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Executes the full feature engineering pipeline sequentially.

        Args:
            df: Cleaned uniform interval DataFrame.
            drop_na: Whether to drop initial warm-up rows (caused by max lag)
                    and terminal rows (caused by horizon targets).

        Returns:
            pd.DataFrame: Engineered feature matrix ready for modeling.
        """
        logger.info("Executing feature engineering pipeline...")
        df_feats = self.add_cyclical_time_features(df)
        df_feats = self.add_autoregressive_lags(df_feats)
        df_feats = self.add_rolling_statistics(df_feats)
        df_feats = self.add_exogenous_calendar_features(df_feats)
        df_feats = self.add_forecast_targets(df_feats)

        if drop_na:
            initial_len = len(df_feats)
            df_feats = df_feats.dropna().reset_index(drop=True)
            logger.info(f"Dropped {initial_len - len(df_feats)} boundary rows (lags & future targets). Remaining: {len(df_feats):,}")

        return df_feats

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Returns the list of engineered feature names, excluding metadata and targets.

        Args:
            df: Processed DataFrame.

        Returns:
            List[str]: Column names to pass to estimators.
        """
        excluded = {
            self.timestamp_col,
            self.target_col,
            self.sensor_id_col if self.sensor_id_col else "",
        }
        for h in self.horizon_steps:
            excluded.add(f"target_h{h}")

        return [col for col in df.columns if col not in excluded]

