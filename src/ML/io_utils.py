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
    n_filas_nan = df.isna().any(axis=1).sum()
    print(f"Number of rows with at least one NaN: {n_filas_nan}")

    df_limpio = df.dropna().copy()


    return df_limpio

import pandas as pd

def load_data(csv_path, target_name, exclude_cols=None):
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    df= remove_nan(df)
    df = df[df["covas_mean"] > 0].copy()
    if target_name not in df.columns:
        raise ValueError(f"La columna target '{target_name}' no está en el CSV.")

    exclude_cols = exclude_cols or []
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_name]]

    print(f"New shape: {df.shape}")
    X = df[feature_cols].copy()
    y = df[target_name].copy()

    return X, y