import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from catboost import CatBoostRegressor

target = 'order'

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
def predict(df,predict_date):
    # 資料前處理
    df, train_data, test_data = data_Preprocessing(df)
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
    predictions = stacked_model.predict(test_data[0])
    mae = mean_absolute_error(test_data[1], predictions)
    print(f"Mean Absolute Error: {mae}")

    target_year = predict_date.year
    target_month = predict_date.month
    month_sin = np.sin(2 * np.pi * target_month / 12)
    month_cos = np.cos(2 * np.pi * target_month / 12)
    target_data = {
    "year": [target_year],
    "month": [target_month],
    "month_sin": [month_sin],
    "month_cos": [month_cos]
    }
    feature_data = {key: target_data[key] for key in feature_col(df) if key in target_data}
    feature_data = pd.DataFrame(feature_data)
    # Make predictions and evaluate
    predictions = stacked_model.predict(feature_data)

    return predictions

# 資料前處理
def data_Preprocessing(df):
    # Extract year and month as separate columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Add a column for the cyclical transformation of the month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Split each class dataset into train and test sets (8:2 ratio)
    # train, test = train_test_split(data[features_high_corr], test_size=0.2, random_state=42, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(df[feature_col(df)], df[target], test_size=0.2, random_state=42, shuffle=True)
    
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return df, [X_train,y_train], [X_test,y_test]

# 特徵提取
def feature_col(df):

    df[target] = df[target].astype(float)

    correlation = df.corr()

    features_high_corr = list(correlation[target].abs().sort_values(ascending=False).iloc[1:3].index)

    return features_high_corr

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