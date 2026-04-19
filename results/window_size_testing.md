## 5-second windows, 50% overlap

### Baseline results

| Model    | R²     | MAE    | RMSE   | MSE    |
|----------|--------|--------|--------|--------|
| LR       | 0.154  | 22.336 | 26.388 | 696.343 |
| RF       | 0.416  | 17.978 | 21.925 | 480.713 |
| SVR      | 0.202  | 21.417 | 25.620 | 656.368 |
| XGBoost  | **0.475** | **16.628** | **20.792** | **432.316** |

### Best baseline model
> **XGBoost** achieved the best baseline performance, with the highest **R² = 0.475** and the lowest error metrics among all tested models.

### Tuned XGBoost

- **Best CV R²:** `0.4296`
- **Best parameters:**
  - `n_estimators = 400`
  - `learning_rate = 0.05`
  - `max_depth = 6`
  - `min_child_weight = 2`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `reg_lambda = 5.0`

### Tuning summary
> In this case, hyperparameter tuning did **not improve** over the baseline test performance.  
> The untuned XGBoost baseline (**R² = 0.475**) remained better than the tuned configuration (**CV R² = 0.430**).

## 8-second windows, 50% overlap

### Baseline results

| Model    | R²     | MAE    | RMSE   | MSE     |
|----------|--------|--------|--------|---------|
| LR       | 0.167  | 22.115 | 25.898 | 670.725 |
| RF       | 0.388  | 18.280 | 22.205 | 493.067 |
| SVR      | 0.209  | 21.436 | 25.243 | 637.212 |
| XGBoost  | **0.435** | **16.937** | **21.333** | **455.110** |

### Best baseline model
> **XGBoost** achieved the best baseline performance, with the highest **R² = 0.435** and the lowest error metrics among all tested models.

## 5-second windows, no overlap

### Baseline results

| Model    | R²     | MAE    | RMSE   | MSE     |
|----------|--------|--------|--------|---------|
| LR       | 0.157  | 21.548 | 25.625 | 656.648 |
| RF       | 0.270  | 19.981 | 23.846 | 568.619 |
| SVR      | 0.137  | 21.877 | 25.914 | 671.528 |
| XGBoost  | **0.310** | **19.057** | **23.185** | **537.523** |

### Best baseline model
> **XGBoost** achieved the best baseline performance, with the highest **R² = 0.310** and the lowest error metrics among all tested models.

## 10-second windows, 50% overlap

### Baseline results

| Model    | R²     | MAE    | RMSE   | MSE     |
|----------|--------|--------|--------|---------|
| LR       | 0.147  | 21.088 | 25.090 | 629.522 |
| RF       | 0.355  | 17.799 | 21.823 | 476.258 |
| SVR      | 0.160  | 21.004 | 24.892 | 619.636 |
| XGBoost  | **0.365** | **16.985** | **21.656** | **468.977** |

### Best baseline model
> **XGBoost** achieved the best baseline performance, with the highest **R² = 0.365** and the lowest error metrics among all tested models.