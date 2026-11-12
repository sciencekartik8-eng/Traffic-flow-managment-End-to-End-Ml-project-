import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
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

target_col = "Approach1_Queue"
features = [col for col in df.columns if col != target_col]

# 3. Scaling (Crucial for Neural Networks)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# 4. Sequence Window Generation
def create_sequences(X, y, window_size=4):
    """Reshape tabular data into 3D windows [samples, window_size, features]"""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

window_size = 4 # Look back 1 hour (4 * 15m)
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# 5. Chronological Train/Test Split (70/30)
split_idx = int(len(X_seq) * 0.7)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

# 6. Define LSTM Architecture
class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output from the last time step
        out = self.fc(out[:, -1, :]) 
        return out

model = TrafficLSTM(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Training Loop (Fast 20 Epochs for benchmark)
print(f"Training LSTM on {len(X_train)} samples...")
epochs = 20
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 8. Evaluation
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t).numpy()

# Inverse transform predictions back to original density scale
preds = scaler_y.inverse_transform(preds_scaled).flatten()
actuals = scaler_y.inverse_transform(y_test).flatten()

mae = np.mean(np.abs(actuals - preds))
rmse = np.sqrt(np.mean((actuals - preds)**2))
w_mape = wmape(actuals, preds)
s_mape = smape(actuals, preds)

print("\n--- LSTM Evaluation Results (Stage 3) ---")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"WMAPE (%): {w_mape:.2f}")
print(f"sMAPE (%): {s_mape:.2f}")