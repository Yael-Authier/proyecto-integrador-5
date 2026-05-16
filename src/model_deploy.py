"""
=================================================
API de Deploy del Modelo
=================================================

Objetivo:
---------
Exponer el modelo entrenado mediante una API construida con FastAPI,
permitiendo realizar predicciones de pago a tiempo a partir de datos
de entrada de un cliente.

Actividades principales:
------------------------
1. Carga del modelo entrenado.
2. Definición de la API.
3. Creación de endpoint de prueba.
4. Creación de endpoint de predicción.
5. Preparación para despliegue.

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

from fastapi import FastAPI


# =================================================
# 2. Inicialización de la API
# =================================================

app = FastAPI(
    title="API de Predicción de Pago de Créditos",
    description="API para predecir si un cliente pagará a tiempo un crédito.",
    version="1.0.0"
)


# =================================================
# 3. Endpoint de prueba
# =================================================

@app.get("/")
def home():
    """
    Endpoint inicial para verificar que la API funciona correctamente.
    """
    return {
        "mensaje": "API de predicción de pago de créditos funcionando correctamente"
    }