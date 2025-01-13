from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


def predict(train_data, valid_data, forecast_steps=6, skip_step=0, grid=False):
    """
    使用 ARIMA 對單變量時間序列進行一次性預測 (multi-step)，並回傳第 skip_step 步的預測值。
    
    參數說明:
    ----------
    train_data  : pd.DataFrame / np.ndarray / list
                  訓練資料 (一維時間序列)
    valid_data  : pd.DataFrame / np.ndarray / list
                  驗證資料 (一維時間序列)
    forecast_steps : int
                  一次性要預測的步數，預設 6
    skip_step   : int
                  要回傳預測結果的 index (0-based)，
                  例如 skip_step=0 代表回傳預測的「第一步」,
                       skip_step=5 代表回傳第六步 (若 forecast_steps=6)
    grid        : bool
                 是否要開啟 grid 超參數搜索
    """
    order=(1,1,1)

    if grid == False:
        # 建立 ARIMA 模型並擬合
        model = ARIMA(train_data[1], order=order)
        results = model.fit()

        # 一次性預測 forecast_steps 步
        forecast = results.forecast(steps=forecast_steps)  # 回傳 shape: (forecast_steps, )

        # 計算 MAE (以 valid_data 的前 forecast_steps 個點為對應)
        #    若 valid_data 長度不足 forecast_steps，則自動截斷
        compare_len = min(forecast_steps, len(valid_data[1]))
        mae = mean_absolute_error(valid_data[1][:compare_len], forecast[:compare_len])

        # 判斷 skip_step 是否在合理範圍，若超過則回傳最後一步
        if skip_step < 0 or skip_step >= forecast_steps:
            skip_step = forecast_steps - 1

        # 印出指定的預測值
        print(f"results.forecast[{skip_step}]: {forecast.values[skip_step]}")

        # 回傳單一步的預測值 (float)
        return float(forecast.values[skip_step])
    else:
        return predict_with_grid_search(train_data, valid_data, forecast_steps=6, skip_step=0)

def predict_with_grid_search(train_data, valid_data, forecast_steps=6, skip_step=0):
    """
    用 grid search 方式遍歷多組 (p, d, q) 參數，找出讓驗證集 MAE 最低的組合。
    最後用該最佳參數輸出最終預測。
    """
    p_values=[0,1,2]
    d_values=[0,1]
    q_values=[0,1,2]
    best_order = None
    best_mae = float('inf')  # 先將最佳 MAE 設成無限大

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    # 1) 建立並訓練 ARIMA
                    model = ARIMA(train_data[1], order=order)
                    results = model.fit()

                    # 2) 多步預測
                    forecast = results.forecast(steps=forecast_steps)

                    # 3) 計算 MAE
                    compare_len = min(forecast_steps, len(valid_data[1]))
                    mae = mean_absolute_error(valid_data[1][:compare_len], forecast[:compare_len])

                    # 4) 若 MAE 更低，更新最佳參數
                    if mae < best_mae:
                        best_mae = mae
                        best_order = order

                except Exception as e:
                    # 某些 (p,d,q) 可能出現模型失敗或不可逆, 直接跳過
                    # 若要除錯可以 print(e)，或 pass
                    pass

    # 用最佳參數建立最終模型
    model = ARIMA(train_data[1], order=best_order)
    results = model.fit()
    forecast = results.forecast(steps=forecast_steps)
    compare_len = min(forecast_steps, len(valid_data[1]))
    final_mae = mean_absolute_error(valid_data[1][:compare_len], forecast[:compare_len])
    print(f"Final MAE with best order: {final_mae:.4f}")

    # skip_step 超界檢查
    if skip_step < 0 or skip_step >= forecast_steps:
        skip_step = forecast_steps - 1

    # 輸出指定預測值
    print(f"results.forecast[{skip_step}]: {forecast.values[skip_step]}")
    return float(forecast.values[skip_step])
