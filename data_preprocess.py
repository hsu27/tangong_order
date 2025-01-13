import pandas as pd
from io import BytesIO
import numpy as np
from fastapi import UploadFile, HTTPException
from sklearn.preprocessing import StandardScaler

def process_excel(file: UploadFile, required_columns=('date', 'order')):
    """
    處理 Excel 文件，檢查必要欄位並轉換日期格式。
    :param file: 上傳的 Excel 文件
    :param required_columns: 必須包含的欄位
    :return: 處理後的 DataFrame
    """
    try:
        file_content = file.file.read()
        excel_data = BytesIO(file_content)
        df = pd.read_excel(excel_data)

        # 檢查必要欄位
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"資料表缺少必要欄位: {', '.join(missing_columns)}")

        # 轉換日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel 文件處理失敗: {e}")

def preprocess_data(df, remove_last=3, column='order'):
    """
    通用的數據處理函數，包括去除最後幾筆資料和標準化。
    :param df: 輸入 DataFrame
    :param remove_last: 要去除的資料筆數
    :param column: 要處理的欄位名稱
    :return: 處理後的 DataFrame, 標準化數據, Scaler
    """
    if remove_last > 0:
        df = df[:-remove_last]

    # 獲取數據並進行標準化
    data = df[column].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return df, scaled_data, scaler

def split_data(data, test_volume=10):
    """
    將數據分割為訓練集和驗證集。
    :param data: 輸入數據（1D 或 2D）
    :param test_volume: 驗證集大小
    :return: 訓練集, 驗證集
    """
    train_data = data[:-test_volume]
    valid_data = data[-test_volume:]
    return train_data, valid_data

def extract_last_records(df, column_date='date', column_value='order', tail=3):
    """
    提取最後 n 筆資料，並格式化為字典列表。
    :param df: 輸入 DataFrame
    :param column_date: 日期欄位名稱
    :param column_value: 值欄位名稱
    :param n: 要提取的筆數
    :return: 最後 n 筆資料的字典列表
    """
    last_records = df[[column_date, column_value]].tail(tail)
    return [
        {"date": record[column_date].strftime("%Y-%m"), "value": record[column_value]}
        for record in last_records.to_dict(orient="records")
    ]

def create_dataset(data, time_step, forecast_horizon, skip_step):
    """
    構建訓練數據集 (X, Y)。
    :param data: 原始數據
    :param time_step: 時間步數
    :param forecast_horizon: 預測範圍
    :param skip_step: 跳過的步數
    :return: 特徵 X 和目標 Y
    """
    X, Y = [], []
    for i in range(len(data) - time_step - skip_step - forecast_horizon + 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step + skip_step:i + time_step + skip_step + forecast_horizon, 0])
    return np.array(X), np.array(Y)