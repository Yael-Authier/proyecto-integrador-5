# Proyecto Integrador MLOps - Predicción de Pago de Créditos

## Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un pipeline completo de Machine Learning y MLOps para predecir si un cliente pagará a tiempo un crédito.

El trabajo incluye:

- Análisis exploratorio de datos (EDA)
- Limpieza y transformación de datos
- Entrenamiento y evaluación de modelos
- Balanceo de clases con SMOTE
- Optimización de hiperparámetros
- Monitoreo de data drift
- Desarrollo de dashboard interactivo con Streamlit
- API de predicción con FastAPI

---

## Estructura del proyecto

```text
PI_M5_Yael_Authier/

├── app/
│   └── streamlit_app.py
│
├── data/
│   └── Base_de_datos.xlsx
│
├── models/
│   └── modelo_xgboost.pkl
│
├── notebooks/
│   ├── cargar_datos.ipynb
│   └── comprension_eda.ipynb
│
├── src/
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   └── model_deploy.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Streamlit
- FastAPI
- Uvicorn
- Joblib

---

## Modelos entrenados

Se evaluaron distintos modelos supervisados:

- Regresión Logística
- Random Forest
- XGBoost

El modelo seleccionado fue XGBoost por obtener el mejor desempeño general.

---

## Resultados principales

### ROC-AUC en test

- Regresión Logística: ~0.58
- Random Forest: ~0.63
- XGBoost: ~0.66

### Validación cruzada optimizada

El modelo XGBoost optimizado alcanzó aproximadamente:

```text
ROC-AUC ≈ 0.978
```

durante validación cruzada sobre datos balanceados mediante SMOTE.

---

## Monitoreo de Data Drift

Se implementó monitoreo utilizando KS Test para detectar cambios estadísticos entre datos históricos y datos actuales simulados.

Actualmente no se detectó drift significativo.

---

## Dashboard Streamlit

El proyecto incluye una aplicación interactiva desarrollada con Streamlit que permite visualizar:

- Información general del dataset
- Métricas del modelo
- Estado de monitoreo
- Variables más importantes

### Ejecutar Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## API con FastAPI

El proyecto incluye una API desarrollada con FastAPI para exponer el modelo entrenado y permitir predicciones en tiempo real.

### Levantar la API localmente

```bash
uvicorn src.model_deploy:app --reload
```

### Documentación automática

Una vez levantada la API, se puede acceder a la documentación Swagger desde:

```text
http://127.0.0.1:8000/docs
```

### Endpoint principal

```text
POST /predict
```

Este endpoint recibe datos de entrada de un cliente y devuelve:

- Predicción del modelo
- Resultado interpretado
- Probabilidad estimada de pago a tiempo

Ejemplo de respuesta:

```json
{
  "prediccion": 0,
  "resultado": "Riesgo de no pago",
  "probabilidad_pago_a_tiempo": 0.2235
}
```

---

## Instalación

### Clonar repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
```

### Crear entorno virtual

```bash
python -m venv venv
```

### Activar entorno virtual en Windows

```bash
venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```
---

## Docker

El proyecto incluye un `Dockerfile` y un `.dockerignore` para construir una imagen con la API de FastAPI y sus dependencias.

### Construir imagen Docker

```bash
docker build -t pi-m5-riesgo-crediticio .
```

### Ejecutar contenedor

```bash
docker run -p 8000:8000 pi-m5-riesgo-crediticio
```

### Acceder a la API

```text
http://127.0.0.1:8000/docs
```

Nota: para ejecutar estos comandos es necesario tener Docker instalado y activo localmente.
---

## Autor

Yael Authier