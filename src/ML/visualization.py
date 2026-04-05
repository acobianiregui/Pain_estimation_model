#src/ML/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

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