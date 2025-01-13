from sklearn.model_selection import train_test_split, KFold
from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np

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