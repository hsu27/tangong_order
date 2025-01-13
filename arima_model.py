from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools

def predict(train_data, valid_data, TEST_VOLUME, skip_step):
    # 定義參數範圍(他會幫你把各組合測試過一次留最好的)
    # p 是自回歸部分的階數
    # d 是差分的階數
    # q 是移動平均部分的階數

    model = ARIMA(train_data, order=(1, 1, 1))
    results = model.fit()

    # 驗證集測試
    forecast = results.forecast(steps=TEST_VOLUME)
    mse = mean_absolute_error(valid_data, forecast)

    print(mse)
    # 預測
    forecast = results.forecast(steps=TEST_VOLUME) 
    print(forecast)
    print(f'results.forecast[{skip_step}]:{forecast.values[skip_step]}')
    return forecast.values[skip_step]