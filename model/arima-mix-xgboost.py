'''混和ARIMA和XGBoost'''
import itertools
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from data_preprocess import *

class ARIMAXGBoostModel:
    ''' 混和ARIMA和XGBoost模型 '''
    def __init__(self, train_data, valid_data, feature_cols,
        arima_grid_p = [0, 1, 2, 3, 4, 5],
        arima_grid_d = [0, 1, 2, 3, 4, 5],
        arima_grid_q = [0, 1, 2, 3, 4, 5],
        xgboost_grid_max_depth = [3, 5, 7],
        xgboost_grid_eta = [0.1, 0.3, 0.5],
        xgboost_grid_subsample = [0.5, 0.7, 0.9],
        xgboost_grid_colsample_bytree = [0.5, 0.7, 0.9]):
        '''
        Initializes the ARIMAXGBoostModel with specified data and hyperparameters for ARIMA and XGBoost models.

        Args:
            data (pd.DataFrame): A DataFrame containing the time series data with a 'date' index and 'order' values.
            test_volume (int, optional): The number of data points to be used as the test set. Default is 10.
            arima_grid_p (list, optional): List of integers representing the ARIMA model's autoregressive term order.
            arima_grid_d (list, optional): List of integers representing the ARIMA model's differencing term order.
            arima_grid_q (list, optional): List of integers representing the ARIMA model's moving average term order.
            xgboost_grid_max_depth (list, optional): List of integers representing the maximum depth of the XGBoost models.
            xgboost_grid_eta (list, optional): List of floats representing the learning rate of the XGBoost models.
            xgboost_grid_subsample (list, optional): List of floats representing the subsample ratio of the XGBoost models.
            xgboost_grid_colsample_bytree (list, optional): List of floats representing the subsample ratio of columns for each tree in the XGBoost models.

        Attributes:
            data (pd.DataFrame): Processed data with 'order' as float type.
            test_volume (int): Number of data points used for validation.
            train_data (pd.Series): Training data extracted from the input data.
            valid_data (pd.Series): Validation data extracted from the input data.
            is_trained (bool): Flag indicating whether the model has been trained.
            error_message (str or None): Message storing any error that occurs during training.
            arima_pdq (list): List of tuples representing all combinations of ARIMA parameters.
            arima_best_mse (float): Best Mean Squared Error found during ARIMA model training.
            arima_best_pdq (tuple or None): Best ARIMA parameters found during model training.
            arima_model (ARIMA or None): Fitted ARIMA model.
            xgboost_param_grid (dict): Dictionary containing hyperparameter grid for XGBoost.
            xgboost_model (XGBRegressor or None): Fitted XGBoost model.
        '''
        warnings.filterwarnings('ignore')   # 忽略警告

        # ## 待改
        # self.data = data
        # self.data['order'] = self.data['order'].astype(float)
        # # 訓練集和驗證集
        # self.test_volume = test_volume
        # self.train_data = self.data['order'][:-self.test_volume]
        # self.valid_data = self.data['order'][-self.test_volume:]
        self.train_data = train_data
        self.valid_data = valid_data
        self.feature_cols = feature_cols
        
        # 已完成訓練
        self.is_trained = False
        self.error_message = None

        # 定義arima參數
        p = arima_grid_p # p 是自回歸部分的階數
        d = arima_grid_d # d 是差分的階數
        q = arima_grid_q # q 是移動平均部分的階數
        self.arima_pdq = list(itertools.product(p, d, q))
        self.arima_best_mae = float("inf")
        self.arima_best_pdq = None
        self.arima_model = None

        # 定義XGBoost參數
        self.xgboost_param_grid = {
            'max_depth': xgboost_grid_max_depth,
            'eta': xgboost_grid_eta,
            'subsample': xgboost_grid_subsample,
            'colsample_bytree': xgboost_grid_colsample_bytree
        }
        self.xgboost_model = None


    def train_model(self):
        '''訓練ARIMA模型及XGBoost模型'''

        # 訓練 ARIMA 模型
        # 搜尋最好的組合，用MSE來判斷。
        for param in self.arima_pdq:
            try:
                arima_model = ARIMA(self.train_data[1], order=param)
                results = arima_model.fit()

                # 驗證集測試
                forecast = results.forecast(steps=len(self.valid_data[1]))
                mae = mean_absolute_error(self.valid_data[1], forecast)

                # 保留效果最好的
                if mae < self.arima_best_mae:
                    self.arima_best_mae = mae
                    self.arima_best_pdq = param
                print(f'ARIMA{param} - MAE:{mae}')
            except Exception as e:
                print(f'ARIMA{param} - Error: {e}')
                continue
        print(f'最佳參數組合: {self.arima_best_pdq} - MAE: {self.arima_best_mae}')
        # 使用最佳參數訓練
        self.arima_model = ARIMA(self.train_data[1], order=self.arima_best_pdq)
        self.arima_model = self.arima_model.fit()
        # ARIMA 預測
        #arima_forecast = self.arima_model.forecast(steps=self.test_volume)
        # 計算 ARIMA 誤差
        #arima_errors = self.valid_data - arima_forecast

        print(1)
        # 訓練 XGBoost 模型
        # #計算 ARIMA 誤差，並在數據集中添加
        y_train = self.train_data[1] - ARIMA(self.train_data[1], order=self.arima_best_pdq).fit().fittedvalues
        y_test = self.valid_data[1] - ARIMA(self.valid_data[1], order=self.arima_best_pdq).fit().fittedvalues
        
        print(2)
        # # 分割數據集，確保時間順序
        # # GridSearchCV for XGBoost
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
        grid_search = GridSearchCV(estimator=xgb_reg, param_grid=self.xgboost_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(self.train_data[0], y_train)
        # 使用最佳參數進行預測
        self.xgboost_model = grid_search.best_estimator_

        # 已完成訓練
        self.is_trained = True
        self.error_message = None

        mae = mean_absolute_error(y_test, self.xgboost_model.predict(self.valid_data[0]))

        return mae

    def predict_future(self, predict_date):
        '''預測未來 1 個月'''
        FUTURE_STEPS = 1            # 預測未來 3 個月
        # 使用最佳參數進行 ARIMA 預測，延伸到未來3個月
        arima_forecast = self.arima_model.forecast(steps=FUTURE_STEPS)

        # # 將 XGBoost 模型輸出對應的特徵集延伸
        # X_new = pd.DataFrame({
        #     'lag1': self.data['order'].shift(1),
        #     'lag2': self.data['order'].shift(2),
        #     'lag3': self.data['order'].shift(3),
        #     'lag4': self.data['order'].shift(4),
        # })

        # # 準備未來3個月的特徵數據
        # X_new = X_new[-FUTURE_STEPS:].fillna(0)  # 如果有缺失值，這裡用0填補

        feature_data = predict_data(predict_date, self.feature_cols)
        # 使用最佳參數進行 XGBoost 預測誤差的校正
        xgb_errors_pred = self.xgboost_model.predict(feature_data)

        # 計算最終預測值
        final_forecast = arima_forecast + xgb_errors_pred

        return final_forecast
    

def predict(train_data, valid_data, feature_cols, predict_date):

    # 建立模型
    model = ARIMAXGBoostModel(train_data, valid_data, feature_cols)

    # 訓練模型
    mae = model.train_model()

    # 預測未來 3 個月
    future_prediction = model.predict_future(predict_date)

    return future_prediction.item(), mae