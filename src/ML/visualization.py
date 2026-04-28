#src/ML/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.inspection import permutation_importance

from src.ML.config import RANDOM_STATE
### LINEARITY ASSUMPTION
def residual_analysis(y_true, y_pred):
    """
    Performs residual analysis on a given linear regression model.
    This function creates a figure with four subplots:
    1. Histogram of the residuals with a fitted normal distribution
    2. Q-Q plot of the residuals
    3. Residuals vs. fitted values
    4. Residuals vs. case order
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_pred - y_true
    fittedvalues = y_pred
    fig = plt.figure(figsize=(10, 8))
    #First subplot: Histogram of residuals with fitted normal distribution
    ax1 = fig.add_subplot(2, 2, 1)
    mu, std = norm.fit(residuals)
    sns.histplot(residuals, stat="density", kde=True, bins = 'auto', ax=ax1)
    # Plot the fitted normal distribution
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'k', linewidth=2, label = 'Normal distribution')
    ax1.set_title('Histogram of Residuals')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Density')
    ax1.legend()
    #Third subplot: Residuals vs. fitted values
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(fittedvalues, residuals, marker='o', linestyle='', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.set_title('Residuals vs. Fitted Values')
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')  
    #Second subplot: Q-Q plot of residuals (be careful interpretation x and y axis !!)
    ax3 = fig.add_subplot(2, 2, 3)
    sm.qqplot(residuals, line='s', ax=ax3)
    ax3.set_title('Q-Q Plot')
    #Fourth subplot: Residuals vs. case order
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(residuals, marker='o', linestyle='', alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', lw=2)
    ax4.set_title('Residuals vs. Case Order')
    ax4.set_xlabel('Case Order')
    ax4.set_ylabel('Residuals')
    #Tight layout for better spacing between subplots
    plt.tight_layout()
    #Display the plots
    plt.show()
    return
#### GENERAL VISUALIZATION FUNCTIONS
def plot_metric_by_model(results_df, metric="MAE", feature_set=None):
    """
    Plots a bar chart of the specified metric for each model.
    """
    df_plot = results_df.copy()

    if feature_set is not None:
        df_plot = df_plot[df_plot["feature_set"] == feature_set]

    plt.figure(figsize=(8, 5))
    plt.bar(df_plot["Model"], df_plot[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    title = f"{metric} by model"
    if feature_set is not None:
        title += f" ({feature_set})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual(y_true, y_pred, model_name="Model", feature_set="full"):
    """
    Plots the predicted values against the actual values for a given model.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("True COVAS")
    plt.ylabel("Predicted COVAS")
    plt.title(f"Predicted vs Actual - {model_name} ({feature_set})")
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, model_name="Model", feature_set="full"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_true

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, alpha=0.6)
    
    plt.axhline(0) #This is the zero error line!!!!
    
    plt.xlabel("True COVAS")
    plt.ylabel("Residual (Predicted - True)")
    plt.title(f"Residual Plot - {model_name} ({feature_set})")
    
    plt.tight_layout()
    plt.show()
##################################################################################3
#MODEL SPECIFIC VISUALIZATION FUNCTIONS
##LINEAR REGRESSION COEFFICIENTS
def plot_lr_coefficients(pipeline, feature_names):
    if 'pca' in pipeline.named_steps:
        raise ValueError("Cannot plot coefficients against original feature names when PCA is used.")
    model = pipeline.named_steps['model'] 
    coefs = model.coef_

    plt.figure()
    plt.bar(feature_names, coefs)
    plt.xticks(rotation=90)
    plt.title("Linear Regression Coefficients")
    plt.ylabel("Coefficient value")
    plt.show()
### FEATURE IMPORTANCE (RANDOM FOREST)
def plot_feature_importance_rf(pipeline, feature_names,top_n=None):
    if 'pca' in pipeline.named_steps:
        raise ValueError("Cannot plot feature importances vs original names when PCA is used.")
    model = pipeline.named_steps['model'] 
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer or None.")
        indices = indices[:top_n]

    plt.figure()
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=90)
    plt.title("Feature Importance")
    plt.show()
## FEARURE IMPORTANCE (SVR)
def plot_svr_permutation_importance(pipeline, X, y, feature_names, n_repeats=10, random_state=RANDOM_STATE):
    """
    Plots the feature importance of features for an SVR model.
    NOTE: Feature importance for SVR is not directly available, so we use permutation importance as an alternative.
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
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    importances_mean = result.importances_mean
    importances_std = result.importances_std

    indices = np.argsort(importances_mean)[::-1]
    ##PLOTTING PART
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(importances_mean)),
        importances_mean[indices],
        yerr=importances_std[indices]
    )
    plt.xticks(
        range(len(importances_mean)),
        [feature_names[i] for i in indices],
        rotation=90
    )
    plt.ylabel("Permutation importance")
    plt.title("SVR Feature Importance (Permutation Importance)")
    plt.tight_layout()
    plt.show()


def plot_corr_with_target(X, y, figsize=(8, 14)):
    X_clean = X.loc[:, X.nunique(dropna=False) > 1].copy()
    corr_with_y = X_clean.corrwith(y).sort_values()

    plt.figure(figsize=figsize)
    corr_with_y.plot(kind="barh")
    plt.xlabel("Correlation with target")
    plt.title("Feature Correlation with Target")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_corr_with_target2(
    X,
    y,
    top_n=12,
    include_negative=True,
    figsize=(9, 6),
    title="Feature Correlation with Target"
):
    """
    Plot the most relevant feature correlations with the target in a slide-friendly way.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target vector.
    top_n : int, default=12
        Number of strongest positive correlations to show.
        If include_negative=True, also shows the top_n most negative correlations.
    include_negative : bool, default=True
        Whether to include the strongest negative correlations as well.
    figsize : tuple, default=(9, 6)
        Figure size.
    title : str, default="Feature Correlation with Target"
        Plot title.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    # Remove constant columns
    X_clean = X.loc[:, X.nunique(dropna=False) > 1].copy()

    # Compute correlations and remove NaNs
    corr_with_y = X_clean.corrwith(y).dropna()

    if corr_with_y.empty:
        raise ValueError("No valid correlations could be computed.")

    if include_negative:
        top_pos = corr_with_y.sort_values(ascending=False).head(top_n)
        top_neg = corr_with_y.sort_values(ascending=True).head(top_n)
        corr_plot = pd.concat([top_neg, top_pos])

        # Remove duplicates if a feature appears in both selections
        corr_plot = corr_plot[~corr_plot.index.duplicated(keep="first")]
        corr_plot = corr_plot.sort_values()
    else:
        corr_plot = corr_with_y.abs().sort_values(ascending=False).head(top_n).index
        corr_plot = corr_with_y.loc[corr_plot].sort_values()

    plt.figure(figsize=figsize)
    ax = corr_plot.plot(kind="barh")

    # Optional color split for readability
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_plot.values]
    ax.clear()
    ax.barh(corr_plot.index, corr_plot.values, color=colors)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Correlation with target")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.show()