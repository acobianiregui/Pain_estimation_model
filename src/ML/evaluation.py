#src/ML/evaluation.py

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'MSE': mse,
        'R2': r2,
        'MAE': mae
    }

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    #Run pipeline to train model correctly (aka no data leakage)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    return metrics

