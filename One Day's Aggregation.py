from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow"
    r" project\data\external\DelhiTrafficDensityDataset"
)
csv_files = sorted(list(DATA_DIR.glob("**/*.csv")))
first_file = csv_files[0]

print(f"Testing aggregation on: {first_file.name}")
df = pd.read_csv(first_file)

# 1. Convert Epoch to IST Datetime
df["datetime"] = pd.to_datetime(df["EpochTime"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")

# 2. Set as index and drop raw EpochTime
df = df.set_index("datetime").drop(columns=["EpochTime"]).sort_index()

# 3. Check time bounds for the day
print(f"Start time IST: {df.index.min()}")
print(f"End time IST:   {df.index.max()}")
print(f"Total 1-second records: {len(df)}")

# 4. Resample into 15-minute intervals computing mean
df_15min_mean = df.resample("15min").mean()

print(f"\nResampled shape (15-min): {df_15min_mean.shape}")
print("\nFirst 4 aggregated 15-minute rows (Mean):")
print(df_15min_mean.head(4)[["QueueDensity1", "StopDensity1", "QueueDensity6", "StopDensity6"]])