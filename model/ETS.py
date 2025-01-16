import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# 定義 ETS 預測方法
def predict(train_data, valid_data, forecast_steps=6, skip_step=0, seasonal='add', seasonal_periods=12):
    # 構建模型
    model = ExponentialSmoothing(
        train_data[1],
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()

    # 進行預測
    forecast = model.forecast(forecast_steps)

    compare_len = min(forecast_steps, len(valid_data[1]))
    mae = mean_absolute_error(valid_data[1][:compare_len], forecast[:compare_len])

    # 判斷 skip_step 是否在合理範圍，若超過則回傳最後一步
    if skip_step < 0 or skip_step >= forecast_steps:
        skip_step = forecast_steps - 1

    # 印出指定的預測值
    print(f"results.forecast[{skip_step}]: {forecast.values[skip_step]}")

    # 回傳單一步的預測值 (float)
    return float(forecast.values[skip_step]), mae

