from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools

from skopt import gp_minimize
from skopt.space import Integer

# 定義目標函數
# 目標是透過貝葉斯優化尋找使 MSE 最小化的參數組合
def sarima_objective(params, train, valid, TEST_VOLUME, s):
    p, d, q, P, D, Q = params
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=TEST_VOLUME)
        mse = mean_squared_error(valid, forecast)
        return mse
    except:
        return float('inf')  # 若模型無法訓練，則返回極大值

# SARIMA 貝葉斯優化函數
def optimize_sarima_bayesian(train, valid, TEST_VOLUME, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    # 定義搜尋空間
    search_space = [
        Integer(p_range[0], p_range[1], name="p"),
        Integer(d_range[0], d_range[1], name="d"),
        Integer(q_range[0], q_range[1], name="q"),
        Integer(P_range[0], P_range[1], name="P"),
        Integer(D_range[0], D_range[1], name="D"),
        Integer(Q_range[0], Q_range[1], name="Q")
    ]

    # 貝葉斯優化
    res = gp_minimize(
        func=lambda params: sarima_objective(params, train, valid, TEST_VOLUME, s),
        dimensions=search_space,
        n_calls=50,  # 設定優化的迭代次數
        random_state=42
    )

    best_params = res.x
    best_mse = res.fun
    return best_params, best_mse

# 預測函式
def predict(train_data, valid_data, TEST_VOLUME, skip_step):
    # 定義參數範圍
    p_range = (0, 3)
    d_range = (0, 2)
    q_range = (0, 3)
    P_range = (0, 1)
    D_range = (0, 1)
    Q_range = (0, 1)
    s = 12  # 假設季節性為一年，即 12 個月

    # 使用貝葉斯優化搜尋最佳參數
    best_params, best_mse = optimize_sarima_bayesian(train_data, valid_data, TEST_VOLUME, p_range, d_range, q_range, P_range, D_range, Q_range, s)

    p, d, q, P, D, Q = best_params
    print(f'最佳參數組合: SARIMA({p},{d},{q})x({P},{D},{Q},{s}) - MSE: {best_mse}')

    # 使用最佳參數組合建立 SARIMA 模型並訓練
    best_model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    best_model_fit = best_model.fit(disp=False)

    # 預測
    forecast = best_model_fit.forecast(steps=TEST_VOLUME)
    print(forecast)
    print(f'results.forecast[{skip_step}]: {forecast.values[skip_step]}')
    return forecast.values[skip_step]
