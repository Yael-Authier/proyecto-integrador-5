"""
=================================================
Entrenamiento y Evaluación de Modelos
=================================================

Objetivo:
---------
Entrenar modelos supervisados para predecir si un cliente pagará a tiempo.

Actividades principales:
------------------------
1. Carga y preparación de datos.
2. Separación en entrenamiento y prueba.
3. Entrenamiento de modelos base.
4. Evaluación con métricas de clasificación.
5. Selección del mejor modelo.

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

from sklearn.model_selection import train_test_split