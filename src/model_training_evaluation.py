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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV


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

    # Análisis de correlación con la variable objetivo
    correlaciones = df.corr(numeric_only=True)["Pago_atiempo"] \
        .sort_values(ascending=False)

    print("\nCorrelaciones con Pago_atiempo:")
    print(correlaciones)

    # Eliminación temporal de variables de fecha
    if "fecha_prestamo" in X.columns:
        X = X.drop(columns=["fecha_prestamo"])

     # Eliminación de posibles variables con data leakage
    columnas_leakage = [
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "puntaje"
    ]

    X = X.drop(columns=columnas_leakage)

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

    # Balanceo de clases con SMOTE solo sobre entrenamiento
    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nDistribución de clases antes de SMOTE:")
    print(y_train.value_counts())

    print("\nDistribución de clases después de SMOTE:")
    print(y_train_smote.value_counts())

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

    modelo_logistico.fit(X_train_smote, y_train_smote)

    # Predicciones
    y_pred = modelo_logistico.predict(X_test)

    y_pred_proba = modelo_logistico.predict_proba(X_test)[:, 1]

    # Evaluación del modelo
    print("\nReporte de clasificación - Regresión Logística:")

    print(classification_report(y_test, y_pred))

    print("\nROC-AUC - Regresión Logística:")

    print(roc_auc_score(y_test, y_pred_proba))

    # =================================================
    # 4. Entrenamiento de Random Forest
    # =================================================

    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    modelo_rf.fit(X_train_smote, y_train_smote)

    # Predicciones
    y_pred_rf = modelo_rf.predict(X_test)

    y_pred_proba_rf = modelo_rf.predict_proba(X_test)[:, 1]

    # Evaluación del modelo
    print("\nReporte de clasificación - Random Forest:")

    print(classification_report(y_test, y_pred_rf))

    print("\nROC-AUC - Random Forest:")

    print(roc_auc_score(y_test, y_pred_proba_rf))

    # =================================================
    # 5. Entrenamiento de XGBoost
    # =================================================

    modelo_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )

    modelo_xgb.fit(X_train_smote, y_train_smote)

    # Predicciones
    y_pred_xgb = modelo_xgb.predict(X_test)

    y_pred_proba_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

    # Evaluación del modelo
    print("\nReporte de clasificación - XGBoost:")

    print(classification_report(y_test, y_pred_xgb))

    print("\nROC-AUC - XGBoost:")

    print(roc_auc_score(y_test, y_pred_proba_xgb))

    # =================================================
    # 6. Importancia de variables - XGBoost
    # =================================================

    importancia_variables = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo_xgb.feature_importances_
    })

    importancia_variables = importancia_variables.sort_values(
        by="Importancia",
        ascending=False
    )

    print("\nTop variables más importantes:")
    print(importancia_variables.head(10))

    # Gráfico
    plt.figure(figsize=(10, 6))

    plt.barh(
        importancia_variables["Variable"].head(10),
        importancia_variables["Importancia"].head(10)
    )

    plt.xlabel("Importancia")
    plt.ylabel("Variables")
    plt.title("Top 10 variables más importantes - XGBoost")

    plt.gca().invert_yaxis()

    plt.show()

    # =================================================
    # 7. Guardado del modelo entrenado
    # =================================================

    joblib.dump(modelo_xgb, "modelo_xgboost.pkl")

    print("\nModelo XGBoost guardado correctamente")

    # =================================================
    # 8. Matriz de confusión - XGBoost
    # =================================================

    matriz_confusion = confusion_matrix(
        y_test,
        y_pred_xgb
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz_confusion
    )

    disp.plot()

    plt.title("Matriz de Confusión - XGBoost")

    plt.show()


    # =================================================
    # 9. Curva ROC - XGBoost
    # =================================================

    RocCurveDisplay.from_estimator(
        modelo_xgb,
        X_test,
        y_test
    )

    plt.title("Curva ROC - XGBoost")

    plt.show()

    # =================================================
    # 10. Optimización de hiperparámetros - XGBoost
    # =================================================

    parametros_xgb = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1]
    }

    grid_xgb = GridSearchCV(
        estimator=XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ),
        param_grid=parametros_xgb,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    grid_xgb.fit(X_train_smote, y_train_smote)

    print("\nMejores hiperparámetros - XGBoost:")
    print(grid_xgb.best_params_)

    print("\nMejor ROC-AUC en validación cruzada:")
    print(grid_xgb.best_score_)
