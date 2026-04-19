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

def load_data(csv_path, target_name, exclude_cols=None):
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    df= remove_nan(df)
    df = df[df["covas_mean"] > 5].copy()
    if target_name not in df.columns:
        raise ValueError(f"La columna target '{target_name}' no está en el CSV.")

    exclude_cols = exclude_cols or []
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_name]]

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
