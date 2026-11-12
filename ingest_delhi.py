from pathlib import Path
import numpy as np
import pandas as pd


# Paths
DATA_DIR = Path(
    r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow"
    r" project\data\external\DelhiTrafficDensityDataset"
)
OUTPUT_FILE = Path(
    r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow"
    r" project\data\interim\delhi_15min_density.parquet"
)

csv_files = sorted(list(DATA_DIR.glob("**/*.csv")))
print(f"Found {len(csv_files)} daily CSV files to process.")

daily_frames = []

for idx, f in enumerate(csv_files, 1):
  print(f"[{idx}/{len(csv_files)}] Processing: {f.name}...")
  df = pd.read_csv(f)

  # Convert epoch to IST datetime and remove tz awareness for clean modeling
  df["datetime"] = (
      pd.to_datetime(df["EpochTime"], unit="s", utc=True)
      .dt.tz_convert("Asia/Kolkata")
      .dt.tz_localize(None)
  )

  # Index & drop raw epoch
  df = df.set_index("datetime").drop(columns=["EpochTime"]).sort_index()

  # Resample to 15-minute mean
  df_15min = df.resample("15min").mean()

  # Retain only the 60 operational intervals (07:00 to 21:45)
  df_15min = df_15min.between_time("07:00", "21:45")

  daily_frames.append(df_15min)

# Concatenate all days and sort chronologically
full_df = pd.concat(daily_frames).sort_index()

# Approach-level aggregation (mean density across paired cameras)
full_df["Approach1_Queue"] = (
    full_df["QueueDensity1"] + full_df["QueueDensity2"]
) / 2
full_df["Approach1_Stop"] = (full_df["StopDensity1"] + full_df["StopDensity2"]) / 2

full_df["Approach2_Queue"] = (
    full_df["QueueDensity3"] + full_df["QueueDensity4"]
) / 2
full_df["Approach2_Stop"] = (full_df["StopDensity3"] + full_df["StopDensity4"]) / 2

full_df["Approach3_Queue"] = (
    full_df["QueueDensity5"] + full_df["QueueDensity6"]
) / 2
full_df["Approach3_Stop"] = (full_df["StopDensity5"] + full_df["StopDensity6"]) / 2

# Save to interim parquet
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
full_df.to_parquet(OUTPUT_FILE)

print("\n--- Processing Summary ---")
print(f"Saved file to: {OUTPUT_FILE}")
print(f"Full dataset shape: {full_df.shape}")
print(f"Date range: {full_df.index.min()} to {full_df.index.max()}")
print(f"Expected rows: {len(csv_files)} days * 60 intervals = {len(csv_files) * 60}")
print(f"Actual rows: {len(full_df)}")