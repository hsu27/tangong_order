import pandas as pd
from io import BytesIO
import numpy as np
from fastapi import UploadFile, HTTPException
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold


target = 'order'

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

def preprocess_data(df, remove_last=3):
    """
    通用的數據處理函數，包括去除最後幾筆資料和標準化。
    :param df: 輸入 DataFrame
    :param remove_last: 要去除的資料筆數
    :param column: 要處理的欄位名稱
    :return: 處理後的 DataFrame, 標準化數據, Scaler
    """
    if remove_last > 0:
        df = df[:-remove_last]
    # 特徵欄位生成
    df = feature_create(df)

    # 獲取數據並進行標準化
    nonScaled_data = df[target].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(nonScaled_data)

    return df, scaled_data, nonScaled_data, scaler

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

# 資料特徵工程
## 日期轉換為特徵欄位
def feature_create(df):
    # Extract year and month as separate columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Add a column for the cyclical transformation of the month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

## 資料切分
def split_x_y(df):
    # Split each class dataset into train and test sets (8:2 ratio)
    # train, test = train_test_split(data[features_high_corr], test_size=0.2, random_state=42, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(df[feature_col(df)], df[target], test_size=0.2, random_state=42, shuffle=True)
    
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return [X_train,y_train], [X_test,y_test]

## 特徵欄位取相關性最大兩值
def feature_col(df):

    df[target] = df[target].astype(float)

    correlation = df.corr()

    features_high_corr = list(correlation[target].abs().sort_values(ascending=False).iloc[1:3].index)

    return features_high_corr