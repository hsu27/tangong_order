import requests

# FastAPI 伺服器的 URL
url = "http://192.168.22.20:6001/data_access_layer/insert_ord_forcast"

# 要發送的資料
data = [
    {
        "date": "2025-01",
        "weight": 25.5,
        "item_type": "A",
        "cus_code": "C123",
        "mg": "MG01",
        "sp_size": 15.2,
        "sp_size2": 10.5,
        "model": "XG",
        "error": 0.1
    },
    {
        "date": "2025-02",
        "weight": 30.0,
        "item_type": "B",
        "cus_code": "C456",
        "mg": "MG02",
        "sp_size": 20.0,
        "sp_size2": 12.0,
        "model": "LSTM",
        "error": 0.2
    },
    {
        "date": "2025-03",
        "weight": 25.5,
        "item_type": "A",
        "cus_code": "C123",
        "mg": "MG01",
        "sp_size": 15.2,
        "sp_size2": 10.5,
        "model": "ARIMA",
        "error": 0.1
    }
]

# 發送 POST 請求
response = requests.post(url, json=data)

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