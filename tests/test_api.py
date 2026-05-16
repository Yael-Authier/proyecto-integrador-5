"""
=================================================
Tests de la API FastAPI
=================================================

Objetivo:
---------
Validar que los endpoints principales de la API funcionen correctamente.

Se prueban:
1. Endpoint raíz `/`
2. Endpoint de predicción `/predict`

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

from fastapi.testclient import TestClient

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.model_deploy import app


# =================================================
# 2. Cliente de prueba
# =================================================

client = TestClient(app)


# =================================================
# 3. Test del endpoint raíz
# =================================================

def test_home_endpoint():
    """
    Verifica que el endpoint raíz responda correctamente.
    """
    response = client.get("/")

    assert response.status_code == 200

    assert "mensaje" in response.json()


# =================================================
# 4. Test del endpoint de predicción
# =================================================

def test_predict_endpoint():
    """
    Verifica que el endpoint de predicción reciba datos de entrada
    y devuelva una respuesta con predicción, resultado y probabilidad.
    """

    payload = {
        "tipo_credito": 7,
        "capital_prestado": 3692160,
        "plazo_meses": 10,
        "edad_cliente": 42,
        "salario_cliente": 2500000,
        "total_otros_prestamos": 1,
        "cuota_pactada": 120000,
        "puntaje_datacredito": 750,
        "cant_creditosvigentes": 3,
        "huella_consulta": 2,
        "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 0,
        "promedio_ingresos_datacredito": 4500000,
        "saldo_mora_codeudor": 0,
        "tendencia_ingresos_Decreciente": 1,
        "tendencia_ingresos_Estable": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    assert "prediccion" in response.json()

    assert "resultado" in response.json()

    assert "probabilidad_pago_a_tiempo" in response.json()