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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from ft_engineering import (
    cargar_datos,
    limpiar_tendencia_ingresos,
    tratar_valores_nulos,
    transformar_variables_categoricas,
    separar_variables_modelo
)


# =================================================
# 2. Ejecución inicial del pipeline
# =================================================

if __name__ == "__main__":

    # Ruta del dataset
    ruta = "data/Base_de_datos.xlsx"

    # Carga de datos
    df = cargar_datos(ruta)

    # Limpieza de inconsistencias
    df = limpiar_tendencia_ingresos(df)

    # Tratamiento de nulos
    df = tratar_valores_nulos(df)

    # Transformación de categóricas
    df = transformar_variables_categoricas(df)

    # Separación de variables
    X, y = separar_variables_modelo(df)

    # Eliminación temporal de variables de fecha
    if "fecha_prestamo" in X.columns:
        X = X.drop(columns=["fecha_prestamo"])

    print("Dimensiones de X:")
    print(X.shape)

    print("\nDimensiones de y:")
    print(y.shape)

    # Separación train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nDimensiones de entrenamiento:")
    print(X_train.shape)

    print("\nDimensiones de prueba:")
    print(X_test.shape)

    # =================================================
    # 3. Entrenamiento de Regresión Logística
    # =================================================

    modelo_logistico = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    modelo_logistico.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo_logistico.predict(X_test)

    y_pred_proba = modelo_logistico.predict_proba(X_test)[:, 1]

    # Evaluación del modelo
    print("\nReporte de clasificación - Regresión Logística:")

    print(classification_report(y_test, y_pred))

    print("\nROC-AUC - Regresión Logística:")

    print(roc_auc_score(y_test, y_pred_proba))