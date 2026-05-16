"""
=================================================
Feature Engineering y Preprocesamiento
=================================================

Objetivo:
---------
Preparar los datos para el entrenamiento de modelos de Machine Learning.

Actividades principales:
------------------------
1. Carga de datos.
2. Corrección de inconsistencias.
3. Tratamiento de valores nulos.
4. Transformación de variables categóricas.
5. Escalado y normalización.
6. Creación de nuevas variables.
7. Preparación final del dataset para modelado.

Proyecto:
---------
Predicción de pago a tiempo en créditos.

Autor:
------
Yael Authier
"""

# =================================================
# 1. Importación de librerías
# =================================================

import pandas as pd
import numpy as np


# =================================================
# 2. Carga de datos
# =================================================

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga la base de datos desde un archivo Excel.
    """
    df = pd.read_excel(ruta_archivo)

    return df


# =================================================
# 3. Limpieza de inconsistencias
# =================================================

def limpiar_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la variable tendencia_ingresos.

    Durante el EDA se detectó que esta variable mezcla categorías válidas
    con valores numéricos inconsistentes. Por este motivo, se conservan
    solo las categorías esperadas y el resto se reemplaza por NaN.
    """
    df = df.copy()

    categorias_validas = [
        "Creciente",
        "Estable",
        "Decreciente"
    ]

    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(categorias_validas),
        np.nan
    )

    return df


# =================================================
# 4. Tratamiento de valores nulos
# =================================================

def tratar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza imputación básica de valores nulos.

    Estrategia:
    - Variables numéricas: mediana.
    - Variables categóricas: moda.
    """
    df = df.copy()

    # Variables numéricas
    variables_numericas = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for columna in variables_numericas:

        df[columna] = df[columna].fillna(
            df[columna].median()
        )

    # Variables categóricas
    variables_categoricas = df.select_dtypes(
        include=["object"]
    ).columns

    for columna in variables_categoricas:

        df[columna] = df[columna].fillna(
            df[columna].mode()[0]
        )

    return df


# =================================================
# 5. Transformación de variables categóricas
# =================================================

def transformar_variables_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas en variables numéricas
    mediante One Hot Encoding.
    """
    df = df.copy()

    variables_categoricas = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    df = pd.get_dummies(
        df,
        columns=variables_categoricas,
        drop_first=True
    )

    return df

# =================================================
# 6. Separación de variables predictoras y objetivo
# =================================================

def separar_variables_modelo(
    df: pd.DataFrame,
    variable_objetivo: str = "Pago_atiempo"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa el dataset en variables predictoras X
    y variable objetivo y.
    """
    df = df.copy()

    X = df.drop(columns=[variable_objetivo])
    y = df[variable_objetivo]

    return X, y

# =================================================
# 7. Ejecución de prueba del script
# =================================================

if __name__ == "__main__":

    # Ruta del dataset
    ruta = "data/Base_de_datos.xlsx"

    # Carga inicial
    df = cargar_datos(ruta)

    print("Dataset cargado correctamente")
    print(df.shape)

    # Limpieza de inconsistencias
    df = limpiar_tendencia_ingresos(df)

    print("\nValores únicos de tendencia_ingresos después de la limpieza:")
    print(df["tendencia_ingresos"].unique())

    # Tratamiento de nulos
    df = tratar_valores_nulos(df)

    print("\nValores nulos restantes:")
    print(df.isnull().sum().sum())

    # Transformación de categóricas
    df = transformar_variables_categoricas(df)

    print("\nDimensiones después de transformar categóricas:")
    print(df.shape)

    # Separación de variables
    X, y = separar_variables_modelo(df)

    print("\nDimensiones de X:")
    print(X.shape)

    print("\nDimensiones de y:")
    print(y.shape)