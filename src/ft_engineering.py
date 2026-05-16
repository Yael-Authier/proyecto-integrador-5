"""
=================================================
Feature Engineering y Preprocesamiento
=================================================

Objetivo:
---------
En este archivo se realizará la preparación de los datos
para el entrenamiento de modelos de Machine Learning.

Actividades principales:
------------------------
1. Tratamiento de valores nulos.
2. Corrección de inconsistencias.
3. Transformación de variables categóricas.
4. Escalado y normalización.
5. Creación de nuevas variables.
6. Preparación final del dataset para modelado.

Proyecto:
----------
Predicción de pago a tiempo en créditos.

Autor:
------
Yael Authier
"""
import pandas as pd
import numpy as np


def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga la base de datos desde un archivo Excel.

    Parámetros:
    -----------
    ruta_archivo: str
        Ruta donde se encuentra el archivo de datos.

    Retorna:
    --------
    pd.DataFrame
        Dataset cargado en formato DataFrame.
    """
    df = pd.read_excel(ruta_archivo)
    return df

def limpiar_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la variable tendencia_ingresos.

    Esta variable debería contener categorías como:
    - Creciente
    - Estable
    - Decreciente

    Durante el EDA se detectaron valores numéricos mezclados,
    por lo que se reemplazan por NaN para tratarlos luego.
    """
    df = df.copy()

    categorias_validas = ["Creciente", "Estable", "Decreciente"]

    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(categorias_validas),
        np.nan
    )

    return df

if __name__ == "__main__":

    # Carga de datos
    ruta = "data/Base_de_datos.xlsx"

    df = cargar_datos(ruta)

    print("Dataset cargado correctamente")
    print(df.shape)

    # Limpieza de tendencia_ingresos
    df = limpiar_tendencia_ingresos(df)

    print("\nValores únicos de tendencia_ingresos:")
    print(df["tendencia_ingresos"].unique())