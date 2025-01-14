from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import importlib
import os
import numpy as np
from dateutil.relativedelta import relativedelta
from fastapi.middleware.cors import CORSMiddleware
import sys
from data_preprocess import *

app = FastAPI()
# uvicorn main:app --reload
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

# HTML 表單頁面
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# 預測端點
@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request, file: UploadFile = File(...), model: str = Form(...)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="上傳的檔案必須為 Excel 格式 (.xlsx)")
    
    try:
        # 處理 Excel 資料並縮放
        df = process_excel(file)

        df, scaled_data, nonScaled_data, scaler = preprocess_data(df, 3)

        # 訓練集跟驗證集
        # TEST_VOLUME = 10 # 從資料集拿幾筆出來做驗證
        # train_data, valid_data = split_data(df['order'], TEST_VOLUME)
        train_data, valid_data = split_x_y(df)

        # 預測結果字典
        result_data = []
        # mae 比較
        best_mae = float('inf')  # 初始設為無窮大，便於比較

        # 取原始資料的最後三筆資料
        last_three_records = extract_last_records(df, tail=3)
        # 定義導入搜尋字典
        model_module = dict()
        for model_type in os.listdir("./model"):
            if model_type.endswith(".py"):
                model_module[(model_type.removesuffix(".py")).split('_')[0]] = importlib.import_module(f".{model_type.removesuffix('.py')}", package="model")
        # 載入模型
        if 'grid' in model:
            grid = True
        else :
            grid = False

        # 分別計算 skip_step = 0, 1, 2 的結果
        for skip_step in range(3):

            # 各模型傳入參數
            model_params = {
                "xgboost": (train_data, valid_data, scaler),   # ok
                "lstm": (train_data, valid_data, scaler, time_step, forecast_horizon, EPOCH, BATCH_SIZE),   # ok
                "arima": (train_data, valid_data, forecast_horizon, skip_step, grid),   # ok
                "sarima": (train_data, valid_data, forecast_horizon, skip_step),   # ok
                "stacking": (train_data, valid_data, feature_col(df), df['date'].iloc[-1] + relativedelta(months=skip_step + 1)),   # ok
                "tabnet": (train_data, valid_data, feature_col(df), df['date'].iloc[-1] + relativedelta(months=skip_step + 1))   # ok
                # DeepAR
                # NGBoost 算誤差區間
            }

            if 'all' in model:
                grid = True
                mae = float('inf')
                predicted_value = dict()
                for model_type in model_module:
                    if model_type in model_params:
                        raw_predict, mae = model_module[model_type].predict(*model_params[model_type])
                        if isinstance(raw_predict, np.ndarray):  # 如果是 array，提取第一個元素
                            predicted_value[model_type] = raw_predict.item()
                        else:
                            predicted_value[model_type] = raw_predict
                        if mae < best_mae:
                            best_mae = mae
                            best_model = model_type

            # 動態導入模型並進行預測
            elif model in model_params:
                # print(*model_params[model])
                predicted_value = model_module.predict(*model_params[model])
                if isinstance(predicted_value, np.ndarray):  # 如果是 array，提取第一個元素
                    predicted_value = predicted_value.item()
            elif grid == True:
                if model.split('_')[0] in model_params:
                    predicted_value = model_module.predict(*model_params[model.split('_')[0]])
            else:
                raise ValueError("Unknown model type")

            # 預測日期為原始數據最後一筆日期後
            predict_date = df['date'].iloc[-1] + relativedelta(months=skip_step + 1)  

            result_data.append({"date": predict_date.strftime("%Y-%m"), "best_model": best_model, "value": predicted_value[best_model]})

        print(type(result_data))
        print(result_data)
        # 返回 JSON 格式的結果結構
        return JSONResponse(content=result_data)
    
    except ImportError  as e:
        raise HTTPException(status_code=500, detail=f"無法載入所選模型 '{model}_model'。請確認模型名稱正確且檔案存在。詳細錯誤：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理文件時出錯: {e}")
