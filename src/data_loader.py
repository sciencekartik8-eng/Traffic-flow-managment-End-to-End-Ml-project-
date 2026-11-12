"""
Data ingestion, cleaning, and resampling pipeline for urban traffic camera datasets.

Handles high-frequency timestamped vehicle counts, reindexes to continuous
temporal grids, prevents forward-looking leakage, and performs chronological splits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging
import pandas as pd
import numpy as np

from src.utils import setup_logger

logger = setup_logger("data_loader")


@dataclass
class DatasetSplits:
    """Container holding chronological train, validation, and test DataFrames."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TrafficDataLoader:
    """Robust time-series ingestion and resampling engine for traffic camera data.

    Attributes:
        timestamp_col: Name of column containing datetime values.
        target_col: Name of column containing traffic volume/counts.
        sensor_id_col: Column indicating individual camera/junction IDs.
        resample_interval: Pandas offset alias string (e.g. '15min', '15T', '1H').
        aggregation_func: Method to aggregate raw events ('sum' for counts, 'mean' for speed).
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        target_col: str = "vehicle_count",
        sensor_id_col: Optional[str] = "junction_id",
        resample_interval: str = "15min",
        aggregation_func: str = "sum",
    ) -> None:
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.sensor_id_col = sensor_id_col
        self.resample_interval = resample_interval
        self.aggregation_func = aggregation_func

    def load_raw_data(
        self,
        filepath: str,
        parse_dates: bool = True,
        sort: bool = True,
    ) -> pd.DataFrame:
        """Loads raw tabular dataset from CSV or Parquet.

        Args:
            filepath: Path to the raw data file.
            parse_dates: Whether to automatically parse the timestamp column.
            sort: Whether to enforce chronological ordering upon load.

        Returns:
            pd.DataFrame: Cleaned raw data.
        """
        logger.info(f"Loading raw traffic data from: {filepath}")
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".parquet", ".pq")):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}. Expected CSV or Parquet.")

        if self.timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' not found in dataset columns: {df.columns.tolist()}")

        if parse_dates:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
            initial_count = len(df)
            df = df.dropna(subset=[self.timestamp_col])
            dropped_rows = initial_count - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with unparseable timestamps.")

        if sort:
            sort_keys = [self.timestamp_col]
            if self.sensor_id_col and self.sensor_id_col in df.columns:
                sort_keys = [self.sensor_id_col, self.timestamp_col]
            df = df.sort_values(by=sort_keys).reset_index(drop=True)

        logger.info(f"Loaded {len(df):,} records successfully.")
        return df

    def resample_sensor_series(
        self,
        df: pd.DataFrame,
        fill_missing: str = "zero",
    ) -> pd.DataFrame:
        """Aggregates raw camera detections into regular intervals for a single sensor.

        Creates an uninterrupted date_range grid between the minimum and maximum
        timestamps to ensure zero missing step periods without leakage.

        Args:
            df: Single-sensor DataFrame with a parsed timestamp column.
            fill_missing: Imputation strategy for intervals with no camera records:
                          'zero' (default for count), 'ffill', or 'linear'.

        Returns:
            pd.DataFrame: Regularly spaced time-series indexed by interval timestamp.
        """
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Floor timestamps to interval start
        df = df.copy()
        df["interval_start"] = df[self.timestamp_col].dt.floor(self.resample_interval)

        agg_dict: Dict[str, str] = {self.target_col: self.aggregation_func}
        # Retain extra numeric features if present (e.g., average speed)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != self.target_col:
                agg_dict[col] = "mean"

        aggregated = df.groupby("interval_start").agg(agg_dict).reset_index()
        aggregated = aggregated.rename(columns={"interval_start": self.timestamp_col})

        # Enforce complete, gapless datetime grid
        min_time = aggregated[self.timestamp_col].min()
        max_time = aggregated[self.timestamp_col].max()
        full_grid = pd.date_range(start=min_time, end=max_time, freq=self.resample_interval, name=self.timestamp_col)
        
        resampled = (
            aggregated.set_index(self.timestamp_col)
            .reindex(full_grid)
            .rename_axis(self.timestamp_col)
            .reset_index()
        )

        # Impute intervals with no detections
        if fill_missing == "zero":
            resampled[self.target_col] = resampled[self.target_col].fillna(0.0)
            resampled = resampled.ffill().bfill()  # ffill other exogenous signals if present
        elif fill_missing == "ffill":
            resampled = resampled.ffill().bfill()
        elif fill_missing == "linear":
            resampled[self.target_col] = resampled[self.target_col].interpolate(method="linear")
            resampled = resampled.ffill().bfill()
        else:
            raise ValueError(f"Unknown fill_missing option: '{fill_missing}'")

        return resampled

    def resample_all(
        self,
        df: pd.DataFrame,
        fill_missing: str = "zero",
    ) -> pd.DataFrame:
        """Processes and resamples high-frequency observations across all junctions.

        Args:
            df: Multi-sensor raw observations.
            fill_missing: Missing interval imputation rule.

        Returns:
            pd.DataFrame: Stacked resampled dataset with consistent temporal step.
        """
        if self.sensor_id_col and self.sensor_id_col in df.columns:
            logger.info(f"Resampling across unique sensors in '{self.sensor_id_col}'...")
            processed_dfs: List[pd.DataFrame] = []
            for sensor_id, group in df.groupby(self.sensor_id_col):
                sensor_resampled = self.resample_sensor_series(group, fill_missing=fill_missing)
                sensor_resampled[self.sensor_id_col] = sensor_id
                processed_dfs.append(sensor_resampled)
            combined = pd.concat(processed_dfs, ignore_index=True)
            logger.info(f"Resampled {len(df):,} events into {len(combined):,} uniform intervals.")
            return combined
        else:
            logger.info("Resampling single time series...")
            return self.resample_sensor_series(df, fill_missing=fill_missing)

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        timestamp_col: str = "timestamp",
        purge_gap_steps: int = 0,
    ) -> DatasetSplits:
        """Splits time-series data chronologically strictly preserving temporal order.

        No data from the future is leaked into the past. An optional purge gap
        can discard transition intervals between splits to guarantee lag independence.

        Args:
            df: Cleaned and resampled DataFrame.
            train_ratio: Proportion of earliest historical data for training (e.g. 0.70).
            val_ratio: Proportion of subsequent data for validation (e.g. 0.15).
            test_ratio: Proportion of latest data for testing (e.g. 0.15).
            timestamp_col: Name of timestamp column to base chronological ordering.
            purge_gap_steps: Number of temporal steps to drop between partitions.

        Returns:
            DatasetSplits: Named container holding (train, val, test) DataFrames.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
            )

        unique_times = np.sort(df[timestamp_col].unique())
        total_steps = len(unique_times)

        train_end_idx = int(total_steps * train_ratio)
        val_end_idx = int(total_steps * (train_ratio + val_ratio))

        train_cutoff = unique_times[train_end_idx - 1]
        val_start_cutoff = unique_times[min(train_end_idx + purge_gap_steps, val_end_idx - 1)]
        val_end_cutoff = unique_times[val_end_idx - 1]
        test_start_cutoff = unique_times[min(val_end_idx + purge_gap_steps, total_steps - 1)]

        train_df = df[df[timestamp_col] <= train_cutoff].copy()
        val_df = df[(df[timestamp_col] >= val_start_cutoff) & (df[timestamp_col] <= val_end_cutoff)].copy()
        test_df = df[df[timestamp_col] >= test_start_cutoff].copy()

        logger.info(
            f"Chronological split complete:\n"
            f"  Train : {len(train_df):,} samples ({train_df[timestamp_col].min()} to {train_df[timestamp_col].max()})\n"
            f"  Val   : {len(val_df):,} samples ({val_df[timestamp_col].min()} to {val_df[timestamp_col].max()})\n"
            f"  Test  : {len(test_df):,} samples ({test_df[timestamp_col].min()} to {test_df[timestamp_col].max()})"
        )

        return DatasetSplits(train=train_df, val=val_df, test=test_df)

