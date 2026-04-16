#src/ML/pipelines.py

from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.ML.models import get_linear_regression, get_random_forest, get_svr,get_xgboost
from src.ML.config import RANDOM_STATE

#General pipeline structure
"""
Step 1: Standardize features (SVR is optional tho)
Step 2: Consider PCA (optional)
Step 3: Fit model (Linear Regression, Random Forest, SVR)
"""

def create_LR_pipeline(PCA_n_components=None):
    if PCA_n_components is not None:
        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=PCA_n_components)),
            ('model', get_linear_regression())
        ]
    else:
        steps = [
            ('scaler', StandardScaler()),
            ('model', get_linear_regression())
        ]
    return Pipeline(steps)

def create_RF_pipeline(PCA_n_components=None,**model_params):
    if PCA_n_components is not None:
        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=PCA_n_components)),
            ('model', get_random_forest(**model_params))
        ]
    else: #If no PCA, no need for scaling (RF properties)
        steps = [
            ('model', get_random_forest(**model_params))
        ]
    return Pipeline(steps)

def create_SVR_pipeline(PCA_n_components=None,**model_params):

    if PCA_n_components is not None:
        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=PCA_n_components)),
            ('model', get_svr(**model_params))
        ]
    else:
        steps = [
            ('scaler', StandardScaler()),
            ('model', get_svr(**model_params))
        ]
    return Pipeline(steps)

def create_XGBOOST_pipeline(PCA_n_components=None,**model_params):
    if PCA_n_components is not None:
        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=PCA_n_components)),
            ('model', get_xgboost(**model_params))
        ]
    else: #If no PCA, no need for scaling (RF properties)
        steps = [
            ('model', get_xgboost(**model_params))
        ]
    return Pipeline(steps)