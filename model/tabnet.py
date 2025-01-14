from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
from data_preprocess import *

def data_trans_np(train_data, valid_data):
    X_train = train_data[0].values
    y_train = train_data[1].values.reshape(-1, 1)
    X_test = valid_data[0].values
    y_test = valid_data[1].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test

# 主要接口
def predict(train_data, valid_data, feature_cols, predict_date):
    X_train, y_train, X_test, y_test = data_trans_np(train_data, valid_data)

    model = TabNetRegressor()  #TabNetRegressor()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["test"],
        eval_metric=["mae"],  # 你可以更改評估指標，例如 mae
        max_epochs=500,
        patience=100,
        batch_size=32
    )
    # Make predictions and evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error: {mae}")

    target_year = predict_date.year
    target_month = predict_date.month
    month_sin = np.sin(2 * np.pi * target_month / 12)
    month_cos = np.cos(2 * np.pi * target_month / 12)
    target_data = {
    "year": [target_year],
    "month": [target_month],
    "month_sin": [month_sin],
    "month_cos": [month_cos]
    }
    feature_data = {key: target_data[key] for key in feature_cols if key in target_data}
    feature_data = pd.DataFrame(feature_data).values
    # Make predictions and evaluate
    predictions = model.predict(feature_data)

    return predictions, mae
