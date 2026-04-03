#src/ML/pipelines.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.ML.models import get_linear_regression, get_random_forest, get_svr


#General pipeline structure
"""
Step 1: Standardize features (SVR is optional tho)
Step 2: Consider PCA (not implemented yet)
Step 3: Fit model (Linear Regression, Random Forest, SVR)
"""

def create_LR_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', get_linear_regression())
    ])

def create_RF_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', get_random_forest())
    ])

def create_SVR_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', get_svr())
    ])