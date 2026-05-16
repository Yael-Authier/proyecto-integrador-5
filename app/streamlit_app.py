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