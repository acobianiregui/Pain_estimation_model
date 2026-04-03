#src/ML/models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.ML.config import RANDOM_STATE

#Methods to get models
def get_linear_regression():
    return LinearRegression()

def get_random_forest():
    return RandomForestRegressor(n_estimators=100, 
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1)

def get_svr():
    return SVR()

#Methods to tune models