from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Body, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import importlib
import os
from typing import Optional
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from data_preprocess import *
from data_output import *
from API import post
from API import get_data
import requests
from pydantic import BaseModel
import json
from fastapi.encoders import jsonable_encoder

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

class GetItem(BaseModel):
    result_df: str = None
    item_type: str = None
    cus_code: str = None
    mg: str = None
    sp_size: float = None
    sp_size2: float = None
    model: str = None
    # #
    # result_df: Optional[str] = None
    # item_type: Optional[str] = None
    # cus_code: Optional[str] = None
    # mg: Optional[str] = None
    # sp_size: Optional[float] = None
    # sp_size2: Optional[float] = None
    # model: Optional[str] = None
    

# HTML 表單頁面
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/getdata", response_class=HTMLResponse)
def get_api_data(request: Request):
    try:
        data = get_data.get_data_main()

        # 使用相對路徑呼叫 API
        relative_path = "/predict"
        base_url = "http://localhost:8000"  # 基底 URL（伺服器運行的位址）

        # 合併成完整的 URL
        url = f"{base_url}{relative_path}"
        
        data["model"] = "all_model"
        response = requests.post(url, params=data)
        print(response.status_code)
        response.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理文件時出錯: {e}")
    # return templates.TemplateResponse("upload.html", {"request": request})

# 預測端點
@app.post("/predict", response_class=JSONResponse)
def predict(request: Request, file: Optional[UploadFile] = File(None), data: GetItem = Depends(None)):
# def predict(request: Request, file: Optional[UploadFile] = File(None), model: str = Form(...), data: Optional[dict] = Body(None)):
    json_data = jsonable_encoder(data)
    # print(json_data)
    # print(type(json_data))
    # print(json_data['result_df'])
    # print(json_data['item_type'])
    # print(json_data['cus_code'])
    # print(json_data['mg'])
    # print(json_data['sp_size'])
    # print(json_data['sp_size2'])
    # print(json_data['model'])
    
    try:
        if not file.filename.endswith('.xlsx') and not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="上傳的檔案必須為 Excel 格式 (.xlsx)")
    except:
        if data == "":
            raise HTTPException(status_code=400, detail="請上傳資料或選擇 API 模式")

    # 初始化紀錄檔
    log_df = log_create()

    try:
        if file.filename.endswith('.xlsx'):
            # 處理 Excel 資料
            raw_df = process_excel(file)
        elif file.filename.endswith('.csv'):
            # 從 API 取得 csv 資料
            raw_df = process_csv(file)
        else:
            model = json_data["model"]
            raw_df = pd.read_json(json_data["result_df"], orient="split")  # 還原 DataFrame
            # 提取來自 get_data 的資料
            item_type = json_data["item_type"]
            cus_code = json_data["cus_code"]
            mg = json_data["mg"]
            sp_size = json_data["sp_size"]
            sp_size2 = json_data["sp_size2"]
            # 取得預測日期
            predict_date= post.get_predict_date()

        df, scaled_data, nonScaled_data, scaler = preprocess_data(raw_df, 3)

        # 訓練集跟驗證集
        train_data, valid_data = split_x_y(df)

        # 預測結果字典
        result_data = []
        # mae 比較, 最佳模型保存
        best_mae = float('inf')  # 初始設為無窮大，便於比較
        best_model = None

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

        max_step = 3
        # 分別計算 skip_step = 0, 1, 2 的結果
        for skip_step in range(3):
            # web
            if data == None:
                # 預測日期為原始數據最後一筆日期後
                date = raw_df['date'].iloc[-(max_step-skip_step)]
                true_value = raw_df[raw_df['date'] == date]['order'].iloc[0]
            # api
            else:
                true_value = 0
                date = predict_date[skip_step]

            # 各模型傳入參數
            model_params = {
                "xgboost": (train_data, valid_data, date, feature_col(df)),   # ok
                "lstm": (train_data, valid_data, time_step, forecast_horizon, EPOCH, BATCH_SIZE),   # ok
                "arima": (train_data, valid_data, forecast_horizon, skip_step, grid),   # ok
                "sarima": (train_data, valid_data, forecast_horizon, skip_step, date, feature_col(df)),   # ok
                "stacking": (train_data, valid_data, feature_col(df), date),   # ok
                "tabnet": (train_data, valid_data, feature_col(df), date),   # ok
                "arima-mix-xgboost": (train_data, valid_data, feature_col(df), date),    # ok
                "ETS": (train_data, valid_data, forecast_horizon, skip_step)    # ok
                # DeepAR
                # NGBoost 算誤差區間
            }

            mae = float('inf')
            predicted_value = dict()
            if 'all' in model:
                grid = True
                for model_type in model_module:
                    if model_type in model_params:
                        raw_predict, mae = model_module[model_type].predict(*model_params[model_type])
                        predicted_value[model_type] = raw_predict
                        if mae < best_mae:
                            best_mae = mae
                            best_model = model_type
                    log_df = log_append(log_df, model_type, date.strftime("%Y-%m"), raw_predict, true_value, mae)

            # 動態導入模型並進行預測
            elif model in model_params and grid == True:
                raw_predict, mae = model_module[model].predict(*model_params[model])
                best_model = model
                predicted_value[model] = raw_predict
                log_df = log_append(log_df, model, date.strftime("%Y-%m"), raw_predict, true_value, mae)
            elif model in model_params:
                if model.split('_')[0] in model_params:
                    raw_predict, mae = model_module[model].predict(*model_params[model])
                    best_model = model
                    predicted_value[model] = raw_predict
                    log_df = log_append(log_df, model, date.strftime("%Y-%m"), raw_predict, true_value, mae)
            
            # web
            if data == None:
                result_data.append({"date": date.strftime("%Y-%m"), "best_model": best_model, "value": predicted_value[best_model]})
            # api
            else:
                result_data.append({
                    "date": date.strftime("%Y-%m"),
                    "value": predicted_value[best_model],
                    "item_type": item_type,
                    "cus_code": cus_code,
                    "mg": mg,
                    "sp_size": sp_size,
                    "sp_size2": sp_size2,
                    "best_model": best_model
                })


        log_save(log_df, (file.filename).removesuffix(".xlsx").split('_'))

        
        if data != None:
            # post data to API
            post.post_data(result_data)
        # 返回 JSON 格式的結果結構
        return JSONResponse(content=result_data)
    
    except ImportError  as e:
        raise HTTPException(status_code=500, detail=f"無法載入所選模型 '{model}_model'。請確認模型名稱正確且檔案存在。詳細錯誤：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理文件時出錯: {e}")
