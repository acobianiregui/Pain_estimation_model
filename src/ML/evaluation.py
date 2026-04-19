#src/ML/evaluation.py

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import pandas as pd
from src.ML.config import RANDOM_STATE
import numpy as np

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    #Run pipeline to train model correctly (aka no data leakage)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    return metrics,y_pred

def compute_permutation_importance_df(
    pipeline,
    X,
    y,
    feature_names,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="neg_mean_absolute_error"
):
    """
    Compute permutation importance and return it as a sorted DataFrame.
    """
    if 'pca' in pipeline.named_steps:
        raise ValueError(
            "Permutation importance is not directly interpretable with PCA in the pipeline, "
            "since the model is using principal components instead of the original features."
        )

    result = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df

def get_top_features_from_importance(importance_df, n_top=None, only_positive=False):
    """
    Return top feature names from a permutation importance DataFrame.
    """
    df = importance_df.copy()

    if only_positive:
        df = df[df["importance_mean"] > 0].copy()

    if n_top is not None:
        if not isinstance(n_top, int) or n_top <= 0:
            raise ValueError("n_top must be a positive integer or None.")
        df = df.head(n_top)

    return df["feature"].tolist()

from sklearn.inspection import permutation_importance
import pandas as pd

def get_top_features_xgb(model, X_train, y_train, n_top=20,
                         scoring="neg_mean_absolute_error",
                         n_repeats=10, random_state=42):
    """
    Compute permutation importance for a fitted XGBoost model and
    return the top n_top feature names plus a full importance DataFrame.

    Parameters
    ----------
    model : fitted estimator or fitted pipeline
        Your trained XGBoost model (or pipeline with XGBoost as final step).
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    n_top : int
        Number of top features to return.
    scoring : str
        Scoring metric for permutation importance.
    n_repeats : int
        Number of shuffles per feature.
    random_state : int
        Random seed.

    Returns
    -------
    top_features : list
        Names of the top n_top features.
    importance_df : pd.DataFrame
        DataFrame with feature importances sorted descending.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame so feature names are preserved.")

    if not isinstance(n_top, int) or n_top <= 0:
        raise ValueError("n_top must be a positive integer.")

    result = permutation_importance(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    top_features = importance_df.head(n_top)["feature"].tolist()

    return top_features, importance_df


def select_top_features(X_train, X_test, top_features):
    """
    Keep only the selected top features in X_train and X_test.
    """
    X_train_top = X_train[top_features].copy()
    X_test_top = X_test[top_features].copy()
    return X_train_top, X_test_top


def get_top_features_rf(model, X_train, y_train, n_top=None, scoring="r2"):
    model.fit(X_train, y_train)

    importance_df = compute_permutation_importance_df(
        pipeline=model,
        X=X_train,
        y=y_train,
        feature_names=X_train.columns,
        scoring=scoring
    )

    top_features = get_top_features_from_importance(
        importance_df,
        n_top=n_top,
        only_positive=False
    )

    return top_features, importance_df

def get_top_features_lr(model, X_train, y_train, n_top=None):
    model.fit(X_train, y_train)

    lr_model = model.named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": np.abs(lr_model.coef_)
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    if n_top is not None:
        top_features = importance_df.head(n_top)["feature"].tolist()
    else:
        top_features = importance_df["feature"].tolist()

    return top_features, importance_df