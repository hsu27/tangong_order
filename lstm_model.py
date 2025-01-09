import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def predict(X, Y, scaler, time_step, forecast_horizon, EPOCH, BATCH_SIZE):
    print(f"X.shape:{X.shape}")
    # 將數據 reshape 成 LSTM 所需的 3D 輸入形狀 (樣本數, 時間步長, 特徵數)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 分割訓練集和測試集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    print(f'x_train:{X_train.shape}')
    print(f'X_test:{X_test.shape}')
    print(f'Y_train:{Y_train.shape}')
    print(f'Y_test:{Y_test.shape}')

    # LSTM 模型
    model = Sequential()
    model.add(LSTM(55, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(forecast_horizon))  # 輸出

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練 LSTM 模型
    model.fit(X_train, Y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=0)

    # 使用 LSTM 模型進行預測
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 將預測結果反轉回原始數據的尺度
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train_actual = scaler.inverse_transform(Y_train)
    Y_test_actual = scaler.inverse_transform(Y_test)

    # 計算 MAE 作為評估指標
    train_mae = mean_absolute_error(Y_train_actual, train_predict)
    test_mae = mean_absolute_error(Y_test_actual, test_predict)

    print(f'Train MAE: {train_mae:.4f}')
    print(f'Test MAE: {test_mae:.4f}')

    # 取得樣本外資料的預測值
    last_test_input = X_test[-time_step:]  # 取得 X_test 的最後一組樣本
    last_test_prediction = model.predict(last_test_input)  # 預測最後一組樣本
    last_test_prediction = scaler.inverse_transform(last_test_prediction)  # 反轉縮放至原始尺度

    print(f"last_test_prediction:{last_test_prediction}")
    return float(last_test_prediction[0][0])
