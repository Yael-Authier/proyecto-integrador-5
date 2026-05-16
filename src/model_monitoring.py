"""
=================================================
Monitoreo del Modelo y Data Drift
=================================================

Objetivo:
---------
Implementar procesos de monitoreo para detectar cambios en la distribución
de los datos que puedan afectar el rendimiento del modelo predictivo.

Actividades principales:
------------------------
1. Carga de datos históricos.
2. Simulación de datos actuales.
3. Cálculo de métricas de data drift.
4. Generación de alertas.
5. Preparación de resultados para visualización.

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
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


# =================================================
# 2. Carga de datos
# =================================================

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga la base de datos histórica desde un archivo Excel.
    """
    df = pd.read_excel(ruta_archivo)

    return df

# =================================================
# 3. Simulación de datos actuales
# =================================================

def simular_datos_actuales(
    df: pd.DataFrame,
    frac: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Simula una muestra de datos actuales a partir del dataset histórico.

    En un entorno productivo, estos datos vendrían de nuevas solicitudes
    o registros recientes.
    """
    datos_actuales = df.sample(
        frac=frac,
        random_state=random_state
    )

    return datos_actuales


# =================================================
# 4. Cálculo de data drift con KS Test
# =================================================

def calcular_drift_ks(
    datos_historicos: pd.DataFrame,
    datos_actuales: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula data drift para variables numéricas utilizando KS Test.

    Si p-value < 0.05, se considera posible drift.
    """
    resultados = []

    variables_numericas = datos_historicos.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for variable in variables_numericas:

        historico = datos_historicos[variable].dropna()
        actual = datos_actuales[variable].dropna()

        estadistico, p_value = ks_2samp(
            historico,
            actual
        )

        drift_detectado = p_value < 0.05

        resultados.append({
            "variable": variable,
            "ks_statistic": estadistico,
            "p_value": p_value,
            "drift_detectado": drift_detectado
        })

    resultados_df = pd.DataFrame(resultados)

    return resultados_df


# =================================================
# 5. Ejecución de prueba del monitoreo
# =================================================

if __name__ == "__main__":

    ruta = "data/Base_de_datos.xlsx"

    df_historico = cargar_datos(ruta)

    df_actual = simular_datos_actuales(df_historico)

    resultados_drift = calcular_drift_ks(
        df_historico,
        df_actual
    )

    print("Resultados de data drift con KS Test:")
    print(resultados_drift)
    # Visualización de p-values por variable
    plt.figure(figsize=(12, 6))

    plt.bar(
        resultados_drift["variable"],
        resultados_drift["p_value"]
    )

    plt.axhline(
        y=0.05,
        linestyle="--",
        label="Umbral drift p-value = 0.05"
    )

    plt.xticks(rotation=90)
    plt.xlabel("Variables")
    plt.ylabel("p-value")
    plt.title("Monitoreo de Data Drift - KS Test")
    plt.legend()
    plt.tight_layout()
    plt.show()