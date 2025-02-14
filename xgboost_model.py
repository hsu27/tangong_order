import xgboost as xgb
import numpy as np

def predict(X, Y, scaler):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # 建立並訓練 XGBoost 模型
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    xgboost_model.fit(X_train, Y_train)
    
    # 使用模型進行預測
    last_test_input = X_test[-1:]
    last_test_prediction = xgboost_model.predict(last_test_input)
    last_test_prediction = scaler.inverse_transform(last_test_prediction.reshape(-1, 1))

    print(f"last_test_prediction:{last_test_prediction}")
    return float(last_test_prediction[0][0])
