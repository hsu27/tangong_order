from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import BytesIO
import importlib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from fastapi.middleware.cors import CORSMiddleware
import sys

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 以 CORS 中介軟體來允許應用程式處理跨來源請求
app.add_middleware(
    CORSMiddleware,             # 使用 CORS 中介軟體
    allow_origins=["*"],        # 允許所有來源發送請求
    allow_credentials=True,     # 允許跨來源請求攜帶憑證（如 Cookies）
    allow_methods=["*"],        # 允許所有 HTTP 方法（如 GET、POST、PUT、DELETE）
    allow_headers=["*"],        # 允許所有標頭
)

# 訓練參數設置
time_step = 6
skip_step = 2
forecast_horizon = 1
EPOCH = 500
BATCH_SIZE = 16
TEST_VOLUME = 6 # 從資料集拿幾筆出來做驗證

# 構建訓練數據集 (X, Y) 的函數
def create_dataset(data, time_step, forecast_horizon, skip_step):
    X, Y = [], []
    for i in range(len(data) - time_step - skip_step - forecast_horizon + 1):
        # X: 取 n 筆數據作為輸入 [t, t+1, ..., t+n]
        X.append(data[i:(i + time_step), 0])
        # Y: 預測第 n+skip_step 筆數據 [t+n+skip_step]，跳過 skip_step 筆
        Y.append(data[i + time_step + skip_step:i + time_step + skip_step + forecast_horizon, 0])
    return np.array(X), np.array(Y)

# Excel 資料處理函數
def process_excel(file: UploadFile):
    try:
        file_content = file.file.read()
        excel_data = BytesIO(file_content)
        df = pd.read_excel(excel_data)
        if 'date' not in df.columns or 'order' not in df.columns:
            raise ValueError("資料表必須包含 'date' 和 'order' 欄位")
        
        # 將 'date' 欄位轉換為日期格式
        df['date'] = pd.to_datetime(df['date'])
        # df = df.iloc[:-3]
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel 文件處理失敗: {e}")

# HTML 表單頁面
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# 預測端點
@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request, file: UploadFile = File(...), model: str = Form(...)):
    print(f'model:{model}')
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="上傳的檔案必須為 Excel 格式 (.xlsx)")
    if not os.path.exists(f"{model}_model.py"):
        raise FileNotFoundError(f"檔案 {model}_model.py 不存在，請確認檔案名稱及路徑。")
    
    try:
        # 處理 Excel 資料並縮放
        df = process_excel(file)
        df = df[:-3]
        order_data = df['order'].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(order_data)

        # 訓練集跟驗證集
        TEST_VOLUME = 10 # 從資料集拿幾筆出來做驗證
        train_data = df['order'][:-TEST_VOLUME]  # 訓練集
        valid_data = df['order'][-TEST_VOLUME:]  # 驗證集

        # 預測結果字典
        result_data = []

        # 取原始資料的最後三筆資料
        last_three_records = df[['date', 'order']].tail(3)

        # 使用 append() 將最後三筆資料逐一加入列表
        for record in last_three_records.to_dict(orient="records"):
            result_data.append({"date": record['date'].strftime("%Y-%m"), "value": record['order']})

        # 載入模型
        # print(f"{model}_model")
        model_module = importlib.import_module(f"{model}_model")

        # 分別計算 skip_step = 0, 1, 2 的結果
        for skip_step in range(3):
            # 建立資料集
            X, Y = create_dataset(scaled_data, time_step, forecast_horizon, skip_step)

            # 各模型傳入參數
            model_params = {
                "xgboost": (X, Y, scaler),
                "lstm": (X, Y, scaler, time_step, forecast_horizon, EPOCH, BATCH_SIZE),
                "arima": (train_data, valid_data, TEST_VOLUME, skip_step),
                "sarima": (train_data, valid_data, TEST_VOLUME, skip_step),
                "arima_grid": (train_data, valid_data, TEST_VOLUME, skip_step),
                "sarima_grid": (train_data, valid_data, TEST_VOLUME, skip_step),
            }

            # 動態導入模型並進行預測
            if model in model_params:
                predicted_value = model_module.predict(*model_params[model])
            else:
                raise ValueError("Unknown model type")

            # 預測日期為原始數據最後一筆日期後
            predict_date = df['date'].iloc[-1] + relativedelta(months=skip_step + 1)  

            result_data.append({"date": predict_date.strftime("%Y-%m"), "value": predicted_value})

        # 返回 JSON 格式的結果結構
        return JSONResponse(content=result_data)
    
    except ImportError  as e:
        raise HTTPException(status_code=500, detail=f"無法載入所選模型 '{model}_model'。請確認模型名稱正確且檔案存在。詳細錯誤：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理文件時出錯: {e}")
