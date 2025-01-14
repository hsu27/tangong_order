import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from data_preprocess import *


#=== 1) 定義目標函數 (最小化 MAE) ============================================
def sarima_objective(params, train_data, valid_data, TEST_VOLUME, s):
    """
    params     : (p, d, q, P, D, Q)
    train_data : [X_train_df, y_train_series]
    valid_data : [X_valid_df, y_valid_series]
    TEST_VOLUME: 預測步數
    s          : 季節性 (seasonal_periods)
    """
    p, d, q, P, D, Q = params
    
    try:
        #=== (a) 建立 SARIMAX: endog=目標序列, exog=外生變數 (X)
        model = SARIMAX(
            endog=train_data[1],   # y_train (Series)
            exog=train_data[0],    # X_train (DataFrame)
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        #=== (b) 擬合模型時，可增加 maxiter，並用其他優化器 method (ex: 'powell')
        # results = model.fit(disp=False)  
        results = model.fit(disp=False, maxiter=300, method='powell')

        #=== (c) 多步預測 (steps=TEST_VOLUME)，外生變數要對應 valid X
        forecast = results.forecast(
            steps=len(valid_data[0]), 
            exog=valid_data[0]  # valid X
        )

        #=== (d) 計算 MAE (與 valid y 比較)
        mae = mean_absolute_error(valid_data[1], forecast)

        return mae

    except Exception as e:
        # 若某些參數無法擬合 (不可逆或其他數值問題)，返回極大值
        return float('inf')


#=== 2) SARIMA 貝葉斯優化函式 (Bayesian Optimization) =========================
def optimize_sarima_bayesian(train_data, valid_data, TEST_VOLUME,
                             p_range, d_range, q_range,
                             P_range, D_range, Q_range, s):
    """
    在 (p,d,q,P,D,Q) 的整數區間內進行 Bayes Optimization (gp_minimize)，
    目標函式是最小化 MAE。
    train_data, valid_data 格式: [X_df, y_series]
    """
    # 定義搜尋空間 (整數範圍)
    search_space = [
        Integer(p_range[0], p_range[1], name="p"),
        Integer(d_range[0], d_range[1], name="d"),
        Integer(q_range[0], q_range[1], name="q"),
        Integer(P_range[0], P_range[1], name="P"),
        Integer(D_range[0], D_range[1], name="D"),
        Integer(Q_range[0], Q_range[1], name="Q")
    ]

    # 使用 gp_minimize 進行貝葉斯優化
    res = gp_minimize(
        func=lambda params: sarima_objective(params, train_data, valid_data, TEST_VOLUME, s),
        dimensions=search_space,
        n_calls=50,       # 迭代次數，可自行調整
        random_state=42   # 固定隨機種子
    )

    best_params = res.x   # [best_p, best_d, best_q, best_P, best_D, best_Q]
    best_score = res.fun  # 對應最小的 MAE
    return best_params, best_score


#=== 3) 預測函式 =============================================================
def predict(train_data, valid_data, TEST_VOLUME, skip_step, predict_date, feature_cols):
    """
    train_data = [X_train_df, y_train_series]
    valid_data = [X_valid_df, y_valid_series]

    skip_step: 預測出的序列中，想取第幾步 (0-based)
    """

    #=== (a) 定義參數搜尋範圍 ===
    p_range = (0, 3)
    d_range = (0, 2)
    q_range = (0, 3)
    P_range = (0, 1)
    D_range = (0, 1)
    Q_range = (0, 1)
    s = 12  # 假設季節性為 12 (如月度資料)。若無季節可改 (s=0, 不用季節)

    #=== (b) 貝葉斯優化 (搜尋最小MAE之參數) ===
    best_params, best_mae = optimize_sarima_bayesian(
        train_data,
        valid_data,
        TEST_VOLUME,
        p_range, d_range, q_range,
        P_range, D_range, Q_range,
        s
    )
    
    p, d, q, P, D, Q = best_params
    print(f'最佳參數: SARIMA({p},{d},{q})x({P},{D},{Q},{s})，對應 MAE: {best_mae:.4f}')

    #=== (c) 以最佳參數組合建立 SARIMAX 模型並訓練 ===
    best_model = SARIMAX(
        endog=train_data[1],   # y_train
        exog=train_data[0],    # X_train
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    best_model_fit = best_model.fit(disp=False)  
    # 若需更高收斂，可考慮: best_model.fit(disp=False, maxiter=300, method='powell')

    forecast = best_model_fit.forecast(
        steps=len(valid_data[0]),
        exog=valid_data[0]
    )
    mae = mean_absolute_error(valid_data[1], forecast)

    feature_data = predict_data(predict_date, feature_cols)
    #=== (d) 多步預測 (steps=TEST_VOLUME)，外生變數請傳對應的 X_valid
    forecast = best_model_fit.forecast(
        steps=1,
        exog=feature_data
    )

    #=== (e) 印出完整預測 & 指定步數值 ===
    print("完整多步預測結果：\n", forecast)

    # 防止 skip_step 越界
    if not (0 <= skip_step < TEST_VOLUME):
        skip_step = TEST_VOLUME - 1


    return forecast.values[0], mae
