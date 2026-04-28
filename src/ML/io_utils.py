import pandas as pd

def remove_nan(df):
    """
    Identifica y elimina las filas que contienen al menos un NaN.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    df_limpio : pd.DataFrame
        DataFrame sin filas con NaN.
    """
    find_nans(df)

    df_limpio = df.dropna().copy()


    return df_limpio
import pandas as pd

def load_data(csv_path, target_name, exclude_cols=None, feature_cols=None, covas_threshold=5):
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")

    df = remove_nan(df)

    if covas_threshold is not None:
        if "covas_mean" not in df.columns:
            raise ValueError("Column 'covas_mean' is not in the CSV.")
        df = df[df["covas_mean"] > covas_threshold].copy()

    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' is not in the CSV.")

    exclude_cols = exclude_cols or []

    if feature_cols is not None:
        #explicitly specified features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"These feature columns are not in the CSV: {missing_cols}")

        if target_name in feature_cols:
            raise ValueError(f"Target column '{target_name}' cannot be included in feature_cols.")

        #remove excluded columns if they end up in feature_cols
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    else:
        #all columsns except target and excluded
        feature_cols = [col for col in df.columns if col not in exclude_cols + [target_name]]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected.")

    print(f"New shape: {df.shape}")

    X = df[feature_cols].copy()
    y = df[target_name].copy()

    return X, y

import pandas as pd

def find_nans(df):
    """
    Devuelve y muestra un resumen de NaNs por columna y el total global.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    
    Retorna
    -------
    resumen : pd.DataFrame
        DataFrame con:
        - nan_count: número de NaNs por columna
        - nan_pct: porcentaje de NaNs por columna
    total_nans : int
        Número total de NaNs en todo el DataFrame
    rows_with_nan : int
        Número de filas con al menos un NaN
    """
    nan_count = df.isna().sum()
    nan_pct = df.isna().mean() * 100

    resumen = pd.DataFrame({
        "nan_count": nan_count,
        "nan_pct": nan_pct
    }).sort_values("nan_count", ascending=False)

    total_nans = int(nan_count.sum())
    rows_with_nan = int(df.isna().any(axis=1).sum())

    print("NaNs per column:")
    print(resumen[resumen["nan_count"] > 0])
    print("\nTotal Nans in dataset:", total_nans)
    print("Rows with at least one Nan:", rows_with_nan)
