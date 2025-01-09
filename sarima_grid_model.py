from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools

# 定義 SARIMA 參數搜尋函式
def optimize_sarima(train, valid, TEST_VOLUME, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [s]))
    best_mse = float("inf")
    best_params = None
    best_seasonal_params = None

    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=seasonal_param)
                results = model.fit()
                forecast = results.forecast(steps=TEST_VOLUME)
                mse = mean_squared_error(valid, forecast)
                if mse < best_mse:
                    best_mse = mse
                    best_params = param
                    best_seasonal_params = seasonal_param
                print(f'SARIMA{param}x{seasonal_param} - MSE: {mse}')
            except Exception as e:
                print(f'SARIMA{param}x{seasonal_param} - Error: {e}')
                continue
    return best_params, best_seasonal_params, best_mse

def predict(train_data, valid_data, TEST_VOLUME, skip_step):
    # 定義參數範圍(他會幫你把各組合測試過一次留最好的)
    # p 是自回歸部分的階數
    # d 是差分的階數
    # q 是移動平均部分的階數
    p = range(0, 5)
    d = range(0, 3)
    q = range(0, 5)
    P = range(0, 2)
    D = range(0, 2)
    Q = range(0, 2)
    s = 12  # 假設季節性為一年，即 12 個月

    # 搜尋最佳參數組合
    best_params, best_seasonal_params, best_mse = optimize_sarima(train_data, valid_data, TEST_VOLUME, p, d, q, P, D, Q, s)
    print(f'最佳參數組合: {best_params} x {best_seasonal_params} - MSE: {best_mse}')

    # 使用最佳參數組合建立 SARIMA 模型並訓練
    best_model = SARIMAX(train_data, order=best_params, seasonal_order=best_seasonal_params)
    best_model_fit = best_model.fit()

    # 輸出模型架構
    # print(best_model.summary())

    # 預測
    forecast = best_model_fit.forecast(steps=TEST_VOLUME) 
    print(forecast)
    print(f'results.forecast[{skip_step}]:{forecast.values[skip_step]}')
    return forecast.values[skip_step]