import pandas as pd
import numpy as np
from pathlib import Path

# Paths
INPUT_PATH = Path(r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow project\data\interim\delhi_15min_density.parquet")
OUTPUT_PATH = Path(r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow project\data\processed\delhi_features.parquet")

print("Loading interim data...")
df = pd.read_parquet(INPUT_PATH)

# We will focus our feature engineering on Approach 1 Queue Density first
# You can expand this loop to other approaches later
target_col = "Approach1_Queue"

print("1. Generating Autoregressive Lag Features...")
# Short-term momentum lags (15, 30, and 45 mins ago)
df[f"{target_col}_lag1"] = df[target_col].shift(1)
df[f"{target_col}_lag2"] = df[target_col].shift(2)
df[f"{target_col}_lag3"] = df[target_col].shift(3)

# Seasonal lag (Exact same 15-min slot from the previous operating day)
# Since there are 60 intervals in our 15-hour operating day (07:00 to 21:45)
df[f"{target_col}_lag60"] = df[target_col].shift(60)

print("2. Generating Cyclical Time Features...")
# Extract time components
hour_minute = df.index.hour * 60 + df.index.minute
day_of_week = df.index.dayofweek

# We encode time cyclically using sine/cosine so the model knows 21:45 is near 07:00
# 1440 minutes in a full 24-hour day
df["time_sin"] = np.sin(2 * np.pi * hour_minute / 1440)
df["time_cos"] = np.cos(2 * np.pi * hour_minute / 1440)

# 7 days in a week
df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

print("3. Dropping NaN rows created by lags...")
# The first 60 rows will have NaNs for lag60, so we must drop them
df_clean = df.dropna()

print("Saving processed feature dataset...")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_parquet(OUTPUT_PATH)

print("\n--- Feature Engineering Summary ---")
print(f"Original shape: {df.shape}")
print(f"Final shape (after dropping NaNs): {df_clean.shape}")
print(f"Features created: lag1, lag2, lag3, lag60, time_sin, time_cos, day_sin, day_cos")
