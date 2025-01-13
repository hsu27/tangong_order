虛擬環境套件
``` bash
pip install statsmodels scikit-learn xgboost tensorflow fastapi uvicorn Jinja2 python-multipart openpyxl scikit-optimize lightgbm catboost
```

[Ref](https://medium.com/seaniap/%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B-%E7%B0%A1%E5%96%AE%E6%98%93%E6%87%82-python%E6%96%B0%E6%89%8B%E7%9A%84fastapi%E4%B9%8B%E6%97%85-ebd09dc0167b)

Installation
```python
pip install fastapi uvicorn Jinja2 python-multipart

from fastapi import FastAPI  
  
app = FastAPI()  
  
@app.get("/")  
def read_root():  
return {"Hello": "FastAPI"}
```

```
uvicorn main:app  --reload --host 0.0.0.0 --port 8000
```

測試頁面 : http://{IPv4_address}:8000/upload

寫入結果 API：http://192.168.22.20:6001/data_access_layer/insert_ord_forcast

FASTAPI 說明文件 : http://192.168.22.20:6001/docs#/

寫入資料欄位
![資料欄位說明](https://github.com/user-attachments/assets/4e37746c-a9ca-4227-b28c-ee9247819f8d)


