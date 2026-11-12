import os
from pathlib import Path
import pandas as pd

# Point to your external dataset path
DATA_DIR = Path(
    r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow"
    r" project\data\external\DelhiTrafficDensityDataset"
)

# 1. List directory contents
print(f"Checking directory: {DATA_DIR}")
if not DATA_DIR.exists():
  print("Directory not found! Please check the path.")
else:
  items = list(DATA_DIR.iterdir())
  print(f"Total files/folders found: {len(items)}")

  # Show first 10 items
  for item in items[:10]:
    print(f" - {item.name}")

  # 2. Pick the first CSV or data file to inspect the schema
  data_files = list(DATA_DIR.glob("**/*.csv")) or list(
      DATA_DIR.glob("**/*.txt")
  )
  if data_files:
    sample_file = data_files[0]
    print(f"\n--- Inspecting Sample File: {sample_file.name} ---")
    df_sample = pd.read_csv(sample_file, nrows=5)
    print("\nColumns:", df_sample.columns.tolist())
    print("\nFirst 5 rows:")
    print(df_sample)
