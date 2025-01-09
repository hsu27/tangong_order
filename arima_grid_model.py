from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools

def predict(train_data, valid_data, TEST_VOLUME, skip_step):
    # 定義參數範圍(他會幫你把各組合測試過一次留最好的)
    # p 是自回歸部分的階數
    # d 是差分的階數
    # q 是移動平均部分的階數
    p = range(0, 3)
    d = range(0, 3)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    # 搜尋最好的組合，用MSE來判斷。
    best_mse = float("inf")
    best_params = None  # 最好的參數記錄起來
    for param in pdq:
        try:
            model = ARIMA(train_data, order=param)
            results = model.fit()

            # 驗證集測試
            forecast = results.forecast(steps=TEST_VOLUME)
            mse = mean_squared_error(valid_data, forecast)

            # 保留效果最好的
            if mse < best_mse:
                best_mse = mse
                best_params = param
            print(f'ARIMA{param} - MSE:{mse}')
        except Exception as e:
            print(f'ARIMA{param} - Error: {e}')
            continue

    print(f'最佳參數組合: {best_params} - MSE: {best_mse}')

    # 使用最佳參數
    model = ARIMA(train_data, order=best_params)
    results = model.fit()

    # 預測
    forecast = results.forecast(steps=TEST_VOLUME) 
    print(forecast)
    print(f'results.forecast[{skip_step}]:{forecast.values[skip_step]}')
    return forecast.values[skip_step]