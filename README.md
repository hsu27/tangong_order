# V1 錯誤訊息

檔案 : data_CT12000.xlsx

all
{"detail":"處理文件時出錯: float() argument must be a string or a number, not 'Timestamp'"}

XGBoost
{"detail":"處理文件時出錯: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:date: datetime64[ns]"}

LSTM
{"detail":"處理文件時出錯: float() argument must be a string or a number, not 'Timestamp'"}

ARIMA  有跑出數值，但是都同樣
```json
[
  {
    "date": "2024-07",
    "best_model": "arima",
    "value": 24.6295317651848
  },
  {
    "date": "2024-08",
    "best_model": "arima",
    "value": 24.6295317651848
  },
  {
    "date": "2024-09",
    "best_model": "arima",
    "value": 24.6295317651848
  }
]
```

SARIMA Grid 
{"detail":"處理文件時出錯: Input y contains infinity or a value too large for dtype('float64')."}

ARIMA Grid
{"detail":"處理文件時出錯: 'arima_grid'"}

Stacking
{"detail":"處理文件時出錯: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:date: datetime64[ns]"}

TabNet
{"detail":"處理文件時出錯: float() argument must be a string or a number, not 'Timestamp'"}
