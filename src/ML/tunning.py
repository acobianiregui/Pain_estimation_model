#src/ML/tunning.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from src.ML.pipelines import create_RF_pipeline, create_SVR_pipeline

from sklearn.model_selection import GridSearchCV


def tune_RF(X_train, y_train):
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
    pipeline = create_RF_pipeline()

    param_grid = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': [1.0, 'sqrt', 'log2']
}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search

def tune_SVR(X_train, y_train):
    """
    Receives training data and performs grid search to tune model with best params.
    Hyperparameters to tune:
    - kernel: ['rbf', 'linear', 'poly'] function behavior
    - C: soft margin parameter, controls boundary flexibility (higher C → less flexible)
    - gamma: kernel coefficient for 'rbf' and 'poly' (higher gamma → more influence of single training points)
    - epsilon: margin of tolerance for error 
    - degree: degree of the polynomial kernel
    """
    pipeline = create_SVR_pipeline()

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

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search