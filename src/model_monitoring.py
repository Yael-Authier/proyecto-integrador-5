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