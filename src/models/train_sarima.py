import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# 1. Zero-Safe Metric Functions
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-8)) * 100

# 2. Load Processed Data
INPUT_PATH = Path(r"C:\Users\KARTIK\OneDrive\Desktop\Master folder\Traffic flow project\data\processed\delhi_features.parquet")
df = pd.read_parquet(INPUT_PATH)

# For SARIMA, we only need the univariate target series
series = df["Approach1_Queue"]

# 3. Train/Test Split (70/30)
split_idx = int(len(series) * 0.7)
train, test = series.iloc[:split_idx], series.iloc[split_idx:]

print(f"Training SARIMA on {len(train)} samples...")
# SARIMA Order: (p, d, q) x (P, D, Q, s)
# p=2 (AR lags), d=0 (stationary), q=1 (MA)
# P=1 (Seasonal AR), D=0, Q=1 (Seasonal MA), s=60 (Daily seasonality)
# We restrict to small orders to keep training time manageable
# Pass numpy values to avoid DatetimeIndex frequency mismatch (operating hours skip overnight intervals)
model = SARIMAX(
    train.values, 
    order=(2, 0, 1), 
    seasonal_order=(1, 0, 1, 60),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit the model (This may take a minute or two)
fitted_model = model.fit(disp=False)

print("Forecasting...")
# Forecast the exact length of the test set as numpy array
preds = fitted_model.forecast(steps=len(test))

# 4. Evaluate (using numpy arrays to ensure exact elementwise alignment)
test_vals = test.values
mae = np.mean(np.abs(test_vals - preds))
rmse = np.sqrt(np.mean((test_vals - preds)**2))
w_mape = wmape(test_vals, preds)
s_mape = smape(test_vals, preds)

print("\n--- SARIMA Evaluation Results (Stage 2) ---")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"WMAPE (%): {w_mape:.2f}")
print(f"sMAPE (%): {s_mape:.2f}")