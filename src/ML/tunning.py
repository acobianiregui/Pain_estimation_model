#src/ML/tunning.py
from scipy.stats import randint, uniform
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from src.ML.pipelines import create_RF_pipeline, create_SVR_pipeline,create_XGBOOST_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def tune_RF(X_train, y_train,PCA_n_components=None):
    """
    Receives training data and performs grid search to tune model with best params.
    Hyperparameters to tune:
    - n_estimators: number of trees in the forest
    - max_depth: maximum depth of the tree (None means nodes are expanded until all leaves are pure)
    - min_samples_split: minimum number of samples required to split an internal node
    - min_samples_leaf: minimum number of samples required to be at a leaf node
    - max_features: number of features to consider when looking for the best split 
        - 1.0: all features
        - 'sqrt': square root of the number of features
        - 'log2': logarithm base 2 of the number of features
    """
    if PCA_n_components != None:
        pipeline=create_RF_pipeline(PCA_n_components=PCA_n_components)
    else:
        pipeline = create_RF_pipeline()
    
    #extensive param_grid
    """
    param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': [1.0, 'sqrt', 'log2']
    }
    """
    #Param_grid soft
    param_grid = {
    'model__n_estimators': [100, 200,300],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt','log2']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search

def tune_SVR(X_train, y_train,PCA_n_components=None):
    """
    Receives training data and performs grid search to tune model with best params.
    Hyperparameters to tune:
    - kernel: ['rbf', 'linear', 'poly'] function behavior
    - C: soft margin parameter, controls boundary flexibility (higher C → less flexible)
    - gamma: kernel coefficient for 'rbf' and 'poly' (higher gamma → more influence of single training points)
    - epsilon: margin of tolerance for error 
    - degree: degree of the polynomial kernel
    """
    if PCA_n_components != None:
        pipeline=create_SVR_pipeline(PCA_n_components=PCA_n_components)
    else:
        pipeline = create_SVR_pipeline()
    """
    param_grid = [
        #grid for rbf kernel (non-linear)
        {
            'model__kernel': ['rbf'],
            'model__C': list(np.logspace(-2, 3, 6)),      #0.01 → 1000
            'model__gamma': list(np.logspace(-4, 0, 5)),  #1e-4 → 1
            'model__epsilon': [0.01, 0.1, 0.5, 1]
        },
        #grid for lineal kernel
        {
            'model__kernel': ['linear'],
            'model__C': list(np.logspace(-2, 3, 6)),
            'model__epsilon': [0.01, 0.1, 0.5, 1]
        },
        #grid for polynomial kernel
        {
            'model__kernel': ['poly'],
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto'],
            'model__degree': [2, 3],
            'model__epsilon': [0.01, 0.1, 0.5]
        }
    ]
    """
    param_grid = [
    {
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10,100],
        'model__gamma': ['scale', 'auto'],
        'model__epsilon': [0.1, 0.5]
    },
    {
            'model__kernel': ['poly'],
            'model__C': [0.1, 1, 10,100],
            'model__gamma': ['scale','auto'],
            'model__degree': [2, 3,4],
            'model__epsilon': [0.01, 0.1]
        }
]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search
def tune_XGBOOST(X_train, y_train,PCA_n_components=None,random=False, n_jobs=-1,random_iter=40):
    if PCA_n_components != None:
        pipeline=create_XGBOOST_pipeline(PCA_n_components=PCA_n_components)
    else:
        pipeline = create_XGBOOST_pipeline()
    """
    param_grid = {
    "model__n_estimators": [200, 300, 400],
    "model__learning_rate": [0.03, 0.05],
    "model__max_depth": [4, 5, 6],
    "model__min_child_weight": [1, 2],
    "model__subsample": [0.8, 0.9],
    "model__colsample_bytree": [0.8, 0.9],
    "model__reg_lambda": [3.0, 5.0, 7.0]
    }
    """
    param_grid = {
    "model__n_estimators": [300, 400, 500],
    "model__learning_rate": [0.03, 0.05],
    "model__max_depth": [5, 6],
    "model__min_child_weight": [1, 2],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    "model__reg_lambda": [4.0, 5.0, 6.0]
    }  
    random_grid={
        "model__n_estimators": randint(200, 700),
        "model__learning_rate": uniform(0.02, 0.08),   
        "model__max_depth": randint(4, 8),             
        "model__min_child_weight": randint(1, 5),     
        "model__subsample": uniform(0.75, 0.15),     
        "model__colsample_bytree": uniform(0.75, 0.15),
        "model__reg_lambda": uniform(3.0, 4.0),        
        "model__gamma": uniform(0.0, 0.3)              
    }
    if not random : 
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='rmse',
            n_jobs=n_jobs,
            verbose=3
        )
        grid_search.fit(X_train, y_train)
        return grid_search
    else:
        random_search=RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=random_grid,
            n_iter=random_iter,
            scoring="rmse",
            cv=5,
            verbose=3,
            random_state=42,
            n_jobs=n_jobs
        )
        random_search.fit(X_train,y_train)
        return random_search