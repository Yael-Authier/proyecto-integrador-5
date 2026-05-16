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

from pydantic import BaseModel
from fastapi import FastAPI
import joblib


# =================================================
# 2. Inicialización de la API
# =================================================

app = FastAPI(
    title="API de Predicción de Pago de Créditos",
    description="API para predecir si un cliente pagará a tiempo un crédito.",
    version="1.0.0"
)

# =================================================
# 3. Carga del modelo entrenado
# =================================================

modelo = joblib.load("models/modelo_xgboost.pkl")

# =================================================
# 4. Estructura de datos de entrada
# =================================================

class ClienteInput(BaseModel):

    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: int
    cant_creditosvigentes: int
    huella_consulta: int
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    saldo_mora_codeudor: float
    tendencia_ingresos_Decreciente: int
    tendencia_ingresos_Estable: int

# =================================================
# 5. Endpoint inicial de prueba
# =================================================

@app.get("/")
def home():
    """
    Endpoint inicial para verificar que la API funciona correctamente.
    """
    return {
        "mensaje": "API de predicción de pago de créditos funcionando correctamente"
    }


# =================================================
# 6. Endpoint de predicción
# =================================================

@app.post("/predict")
def predict(cliente: ClienteInput):
    """
    Endpoint para generar una predicción a partir de datos enviados por el usuario.
    """

    datos_cliente = [[
        cliente.tipo_credito,
        cliente.capital_prestado,
        cliente.plazo_meses,
        cliente.edad_cliente,
        cliente.salario_cliente,
        cliente.total_otros_prestamos,
        cliente.cuota_pactada,
        cliente.puntaje_datacredito,
        cliente.cant_creditosvigentes,
        cliente.huella_consulta,
        cliente.creditos_sectorFinanciero,
        cliente.creditos_sectorCooperativo,
        cliente.creditos_sectorReal,
        cliente.promedio_ingresos_datacredito,
        cliente.saldo_mora_codeudor,
        cliente.tendencia_ingresos_Decreciente,
        cliente.tendencia_ingresos_Estable
    ]]

    prediccion = modelo.predict(datos_cliente)

    probabilidad = modelo.predict_proba(datos_cliente)[0][1]

    resultado = "Paga a tiempo" if int(prediccion[0]) == 1 else "Riesgo de no pago"

    return {
        "prediccion": int(prediccion[0]),
        "resultado": resultado,
        "probabilidad_pago_a_tiempo": round(float(probabilidad), 4)
    }