import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from data_preprocess import *

def predict(train_data, valid_data, predict_date, feature_cols):
    # train_size = int(len(X) * 0.8)
    # X_train, X_test = X[:train_size], X[train_size:]
    # Y_train, Y_test = Y[:train_size], Y[train_size:]

    # 建立並訓練 XGBoost 模型
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    xgboost_model.fit(train_data[0], train_data[1])
    
    test_prediction = xgboost_model.predict(valid_data[0])
    mae = mean_absolute_error(valid_data[1], test_prediction)
    # 使用模型進行預測
    # last_test_input = valid_data[0][-1:]
    feature_data = predict_data(predict_date, feature_cols)
    last_test_prediction = xgboost_model.predict(feature_data)
    # last_test_prediction = scaler.inverse_transform(last_test_prediction.reshape(-1, 1))
    
    return float(last_test_prediction[0]), mae
