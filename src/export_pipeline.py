import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

CHECKPOINT_DIR = Path(r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow project\checkpoints")
DATA_PATH = Path(r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow project\data\processed\delhi_features.parquet")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
target_col = "Approach1_Queue"

# Generate 4-step horizons (15m, 30m, 45m, 60m)
for h in range(1, 5):
    df[f"target_h{h}"] = df[target_col].shift(-h)

df = df.dropna()
target_cols = [f"target_h{h}" for h in range(1, 5)]
feature_cols = [c for c in df.columns if not c.startswith("target_h") and c != target_col]

X = df[feature_cols]
Y = df[target_cols]

print("Training production Multi-Horizon XGBoost model...")
models = {}
for h_idx, col in enumerate(target_cols):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, Y[col])
    models[f"h{h_idx + 1}"] = model

# Package pipeline bundle
bundle = {
    "models": models,
    "feature_cols": feature_cols,
    "horizons": [15, 30, 45, 60],
    "target_col": target_col,
    "congestion_thresholds": {
        "free_flow": 0.20,
        "moderate": 0.45,
        "congested": 0.65
    }
}

artifact_path = CHECKPOINT_DIR / "traffic_service_bundle.joblib"
joblib.dump(bundle, artifact_path)
print(f"[SUCCESS] Packaged service bundle saved to: {artifact_path}")