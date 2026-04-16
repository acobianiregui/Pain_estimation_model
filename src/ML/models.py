#src/ML/models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.ML.config import RANDOM_STATE

#Methods to get models
def get_linear_regression():
    return LinearRegression()

def get_random_forest(**model_params):
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    }

    #Override with given parameters (if given)
    params.update(model_params)
    return RandomForestRegressor(random_state=RANDOM_STATE, **params)

def get_svr(**model_params):
    params = {
        "C": 1.0,
        "gamma": 'scale',
        "kernel": 'rbf'
    }
    params.update(model_params)
    return SVR(**params)

def get_xgboost(**model_params):
    params = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0
    }
    params.update(model_params)
    return XGBRegressor(**model_params)
