import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from catboost import CatBoostRegressor
from data_preprocess import *

target = 'order'

# Define base models with parameter tuning using Bayesian Optimization
def bayes_search_model(model, search_spaces, train_data):
    bayes_search = BayesSearchCV(
        model,
        search_spaces,
        n_iter=30,
        cv=5,
        scoring='neg_mean_absolute_error',
        random_state=42,
        verbose=1
    )
    bayes_search.fit(train_data[0], train_data[1])
    print(f"Best parameters for {model.__class__.__name__}: {bayes_search.best_params_}")
    return bayes_search.best_estimator_

# 貝葉斯優化, 使用模型：xgboost, lightgbm, catboost
class ModelTuner:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def bayes_search_model(self, model, search_spaces, n_iter=30, cv=5, scoring='neg_mean_absolute_error'):
        bayes_search = BayesSearchCV(
            model,
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            verbose=1
        )
        bayes_search.fit(self.X_train, self.y_train)
        print(f"Best parameters for {model.__class__.__name__}: {bayes_search.best_params_}")
        return bayes_search.best_estimator_

    def tune_xgb(self):
        xgb_params = {
            'n_estimators': (50, 200, 'log-uniform'),
            'learning_rate': (0.01, 0.5, 'log-uniform'),
            'max_depth': (3, 50, 'log-uniform'),
            'tree_method': ['gpu_hist'],
            'predictor': ['gpu_predictor']
        }
        return self.bayes_search_model(xgb.XGBRegressor(random_state=42), xgb_params)

    def tune_lgb(self):
        lgb_params = {
            'n_estimators': (50, 200, 'log-uniform'),
            'learning_rate': (0.01, 0.5, 'log-uniform'),
            'num_leaves': (31, 70, 'log-uniform'),
            'device_type': ['gpu']
        }
        return self.bayes_search_model(lgb.LGBMRegressor(random_state=42), lgb_params)

    def tune_catboost(self):
        cat_params = {
            'iterations': (50, 200),
            'learning_rate': (0.01, 0.2, 'log-uniform'),
            'depth': (3, 16, 'log-uniform'),
            'l2_leaf_reg': (1, 50, 'log-uniform')
        }
        return self.bayes_search_model(CatBoostRegressor(verbose=0, random_state=42), cat_params)

# StackingRegressor class
class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Create out-of-fold predictions for training meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X.iloc[train_idx], y.iloc[train_idx])
                self.base_models_[i].append(instance)
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X.iloc[holdout_idx])
        
        # Train meta-model
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.mean([model.predict(X) for model in base_models], axis=0)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# 訓練模型
def training(base_models, X_train, y_train):
    # Initialize meta-model and stacking regressor
    meta_model = Lasso(alpha=0.1, random_state=42)
    stacked_model = StackingRegressor(base_models=base_models, meta_model=meta_model)
    # Train the stacked model
    stacked_model.fit(X_train, y_train)

    return stacked_model

# 主要接口
def predict(train_data, valid_data, feature_cols, predict_date):
    # # 資料前處理
    # df = feature_create(df)
    # train_data, test_data = split_x_y(df)

    # 搜尋最佳參數
    tuner = ModelTuner(train_data[0], train_data[1])
    xgb_model = tuner.tune_xgb()
    lgb_model = tuner.tune_lgb()
    cat_model = tuner.tune_catboost()
    
    base_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ]

    # 訓練模型
    stacked_model = training(base_models, train_data[0], train_data[1])

    # Make predictions and evaluate
    predictions = stacked_model.predict(valid_data[0])
    mae = mean_absolute_error(valid_data[1], predictions)

    feature_data = predict_data(predict_date, feature_cols)
    # Make predictions and evaluate
    predictions = stacked_model.predict(feature_data)

    return predictions.item(), mae
