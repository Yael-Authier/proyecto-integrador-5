"""
=================================================
Aplicación Streamlit - Dashboard de Riesgo Crediticio
=================================================

Objetivo:
---------
Construir una aplicación web simple para visualizar información general
del dataset utilizado en el proyecto, mostrar métricas principales y
servir como base para el monitoreo del modelo.

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

import streamlit as st
import pandas as pd


# =================================================
# 2. Configuración general de la aplicación
# =================================================

st.set_page_config(
    page_title="Dashboard Riesgo Crediticio",
    layout="wide"
)


# =================================================
# 3. Título y descripción del proyecto
# =================================================

st.title("Dashboard de Riesgo Crediticio")

st.write(
    """
    Esta aplicación forma parte del Proyecto Integrador del Módulo 5.

    El objetivo del proyecto es desarrollar un modelo predictivo capaz de
    estimar si un cliente pagará a tiempo un crédito, utilizando información
    histórica financiera, crediticia y demográfica.

    En esta primera versión del dashboard se visualiza una descripción general
    del dataset utilizado para el entrenamiento del modelo.
    """
)


# =================================================
# 4. Carga de datos
# =================================================

@st.cache_data
def cargar_datos():
    """
    Carga el dataset original desde la carpeta data.
    """
    df = pd.read_excel("data/Base_de_datos.xlsx")

    return df


df = cargar_datos()


# =================================================
# 5. Métricas generales del dataset
# =================================================

st.subheader("Vista general del dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Cantidad de registros", df.shape[0])

col2.metric("Cantidad de variables", df.shape[1])

col3.metric(
    "Clientes que pagan a tiempo",
    int(df["Pago_atiempo"].sum())
)


# =================================================
# 6. Visualización inicial de datos
# =================================================

st.subheader("Primeras filas del dataset")

st.write(
    """
    A continuación se muestran las primeras filas de la base para verificar
    la estructura de los datos cargados.
    """
)

st.dataframe(df.head())

# =================================================
# 7. Resultados del modelo
# =================================================

st.subheader("Resultados del modelo XGBoost")

st.write(
    """
    Luego de comparar distintos modelos supervisados, se seleccionó XGBoost
    como el modelo con mejor desempeño general.

    El modelo fue entrenado luego de aplicar limpieza de datos, tratamiento
    de valores nulos, transformación de variables categóricas y balanceo de
    clases mediante SMOTE.
    """
)

col4, col5, col6 = st.columns(3)

col4.metric("ROC-AUC Test", "0.66")
col5.metric("Recall clase 0", "0.33")
col6.metric("Modelo seleccionado", "XGBoost")


# =================================================
# 8. Monitoreo de data drift
# =================================================

st.subheader("Monitoreo de Data Drift")

st.write(
    """
    Se implementó un monitoreo inicial de data drift utilizando KS Test.
    En la simulación realizada no se detectaron cambios estadísticamente
    significativos entre los datos históricos y los datos actuales simulados.
    """
)

st.success("Estado actual: no se detecta data drift significativo.")

# =================================================
# 9. Variables más importantes del modelo
# =================================================

st.subheader("Top variables más importantes")

st.write(
    """
    El siguiente gráfico muestra las variables con mayor importancia
    dentro del modelo XGBoost entrenado.
    """
)

variables = [
    "tipo_laboral_Independiente",
    "tipo_credito",
    "creditos_sectorCooperativo",
    "plazo_meses",
    "tendencia_ingresos_Decreciente",
    "puntaje_datacredito",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos_Estable",
    "creditos_sectorFinanciero",
    "total_otros_prestamos"
]

importancias = [
    0.197514,
    0.162944,
    0.092683,
    0.086780,
    0.071880,
    0.063251,
    0.042574,
    0.037592,
    0.036223,
    0.035316
]

importance_df = pd.DataFrame({
    "Variable": variables,
    "Importancia": importancias
})

st.bar_chart(
    importance_df.set_index("Variable")
)

