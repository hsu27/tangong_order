import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=55, hidden_size2=30, output_size=6):
        """一次性預測 6 步，故 output_size=6"""
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size2, output_size)  # output_size=6

    def forward(self, x):
        """
        x shape: (batch_size, time_step, input_size)
        """
        x, _ = self.lstm1(x)           # -> (batch_size, time_step, hidden_size1)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)           # -> (batch_size, time_step, hidden_size2)
        x = self.dropout2(x)
        # 只取最後一個 time_step 進行線性層
        return self.fc(x[:, -1, :])    # -> (batch_size, 6)

def reshape_data(data, num_samples, time_step, pad_value=0.0):
    """
    將資料 reshape 成 (num_samples, time_step, feature_dim)。  
    若不足則用 pad_value 補齊，多了則截斷。
    """
    # 1) 統一轉成 numpy
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = np.array(data, dtype=np.float32)

    # 2) 保證至少 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_features = data.shape
    # 總共需要的元素量
    needed = num_samples * time_step * n_features
    current = n_samples * n_features

    # 補齊或截斷
    if current < needed:
        pad_len = needed - current
        data = np.pad(data.flatten(), (0, pad_len), 'constant', constant_values=pad_value)
        data = data.reshape(-1, n_features)
    elif current > needed:
        data = data.flatten()[:needed].reshape(-1, n_features)

    # 最終形狀 (num_samples, time_step, n_features)
    data = data.reshape(num_samples, time_step, n_features)
    return torch.tensor(data, dtype=torch.float32)

def predict(train_data, valid_data, time_step, forecast_horizon, EPOCH=20, BATCH_SIZE=32):
    """
    一次性預測 6 步：
      - 模型輸出 (output_size=6)
      - Y_train / Y_valid shape -> (num_samples, 6)
    train_data = [X_train, Y_train], valid_data = [X_valid, Y_valid]
    time_step: LSTM 輸入的序列長度
    """

    #==============================
    # 1) 取出資料
    #==============================
    X_train, Y_train = train_data
    X_valid, Y_valid = valid_data

    # DataFrame -> numpy (如有需要)
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(Y_train, pd.DataFrame):
        Y_train = Y_train.values
    if isinstance(X_valid, pd.DataFrame):
        X_valid = X_valid.values
    if isinstance(Y_valid, pd.DataFrame):
        Y_valid = Y_valid.values

    #==============================
    # 2) 計算可分成幾個樣本
    #==============================
    num_samples_train = len(X_train) // time_step
    num_samples_valid = len(X_valid) // time_step

    #==============================
    # 3) Reshape X (padding/truncating)
    #==============================
    X_train = reshape_data(X_train, num_samples_train, time_step)
    X_valid = reshape_data(X_valid, num_samples_valid, time_step)

    #==============================
    # 4) Reshape Y (對應多步)
    #   一次性預測 6 步 -> output_size=6
    #==============================
    forecast_horizon = 6
    # 取對應長度 (num_samples * forecast_horizon)
    Y_train = Y_train[:num_samples_train * forecast_horizon]
    Y_valid = Y_valid[:num_samples_valid * forecast_horizon]
    # reshape 成 (num_samples, 6)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(num_samples_train, forecast_horizon)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32).view(num_samples_valid, forecast_horizon)

    #==============================
    # 5) 轉到 GPU (如可用)
    #==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_valid, Y_valid = X_valid.to(device), Y_valid.to(device)

    #==============================
    # 6) 建 DataLoader (僅 train)
    #==============================
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #==============================
    # 7) 模型、損失、優化器
    #==============================
    model = LSTMModel(
        input_size=X_train.shape[2],   # feature_dim
        hidden_size1=55,
        hidden_size2=30,
        output_size=forecast_horizon   # 6 步
    ).to(device)

    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #==============================
    # 8) 訓練
    #==============================
    model.train()
    for epoch in range(EPOCH):
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()

    #==============================
    # 9) 評估 (MAE)
    #==============================
    model.eval()
    with torch.no_grad():
        valid_pred = model(X_valid)  # (num_samples_valid, 6)

    valid_mae = mean_absolute_error(
        Y_valid.cpu().numpy(), valid_pred.cpu().numpy()
    )
    # print(f"Valid MAE: {valid_mae:.4f}")

    #==============================
    # 10) Out-of-sample 預測範例
    #     這裡示範用 valid_data 最後一筆 X 來預測 6 步
    #==============================
    last_test_input = X_valid[-1:, :, :]  # shape: (1, time_step, feature_dim)
    with torch.no_grad():
        last_test_prediction = model(last_test_input).cpu().numpy()
    print(f"last_test_prediction (6 步): {last_test_prediction[0]}")
    
    # 回傳第一步預測值
    return float(last_test_prediction[0][0]), valid_mae
