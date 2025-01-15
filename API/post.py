import requests
import pandas as pd

# FastAPI 伺服器的 URL
url = "http://192.168.22.20:6001/data_access_layer/insert_ord_forcast"


# 定義預測日期
def get_predict_date():
    date = ["2025-01", "2025-02", "2025-03"]
    return date

# 要發送的資料
data = [
    {
        "date": get_predict_date()[0],
        "weight": 0,
        "item_type": "A",
        "cus_code": "C123",
        "mg": "MG01",
        "sp_size": 15.2,
        "sp_size2": 10.5,
        "model": "",
        "error": 0.1
    },
    {
        "date": get_predict_date()[1],
        "weight": 0,
        "item_type": "B",
        "cus_code": "C456",
        "mg": "MG02",
        "sp_size": 20.0,
        "sp_size2": 12.0,
        "model": "",
        "error": 0.2
    },
    {
        "date": get_predict_date()[2],
        "weight": 0,
        "item_type": "A",
        "cus_code": "C123",
        "mg": "MG01",
        "sp_size": 15.2,
        "sp_size2": 10.5,
        "model": "",
        "error": 0.1
    }
]

def post_data(pred_data):
    # 將資料轉換為 DataFrame
    data_df = pd.DataFrame(data)
    pred_df = pd.DataFrame(pred_data)

    # 根據 date 合併 pred_data 到 data
    merged_df = pd.merge(data_df, pred_df, on='date', suffixes=('', '_pred'))

    # 更新 weight 和 model 欄位
    merged_df['weight'] = merged_df['weight_pred']
    merged_df['model'] = merged_df['model_pred']

    # 刪除多餘的欄位
    merged_df = merged_df.drop(columns=['weight_pred', 'model_pred'])

    data_json = merged_df.to_json(orient="records")

    # 發送 POST 請求
    response = requests.post(url, json=data_json)

    # 檢查回應
    if response.status_code == 200:
        response_body = response.json()
        if response_body.get("result") is True:
            print("資料新增成功！")
        else:
            print("伺服器回應成功，但結果為 false！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}")
        print(response.json())  # 輸出錯誤訊息