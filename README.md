虛擬環境套件
```
pip install statsmodels scikit-learn xgboost tensorflow fastapi uvicorn Jinja2 python-multipart openpyxl
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

uvicorn main:app --reload
啟動應用程式並啟用 **自動重新載入** 功能。

uvicorn main:app  --reload --host 0.0.0.0 --port 8000
啟動應用程式並指定 **主機地址** 和 **埠號**
   
uvicorn main:app --port 8000
啟動應用程式並指定埠號，主機默認為 `127.0.0.1`
