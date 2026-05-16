\# Proyecto Integrador MLOps - Predicción de Pago de Créditos



\## Descripción del proyecto



Este proyecto tiene como objetivo desarrollar un pipeline completo de Machine Learning y MLOps para predecir si un cliente pagará a tiempo un crédito.



El trabajo incluye:



\- Análisis exploratorio de datos (EDA)

\- Limpieza y transformación de datos

\- Entrenamiento y evaluación de modelos

\- Balanceo de clases con SMOTE

\- Optimización de hiperparámetros

\- Monitoreo de data drift

\- Desarrollo de dashboard interactivo con Streamlit



\---



\# Estructura del proyecto



```text

PI\_M5\_Yael\_Authier/



│

├── app/

│   └── streamlit\_app.py

│

├── data/

│   └── Base\_de\_datos.xlsx

│

├── models/

│   └── modelo\_xgboost.pkl

│

├── notebooks/

│   ├── cargar\_datos.ipynb

│   └── comprension\_eda.ipynb

│

├── src/

│   ├── ft\_engineering.py

│   ├── model\_training\_evaluation.py

│   └── model\_monitoring.py

│

├── requirements.txt

├── .gitignore

└── README.md



```

\# Tecnologías utilizadas



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- Imbalanced-learn (SMOTE)

\- Matplotlib

\- Streamlit



\---



\# Modelos entrenados



Se evaluaron distintos modelos supervisados:



\- Regresión Logística

\- Random Forest

\- XGBoost



El modelo seleccionado fue XGBoost por obtener el mejor desempeño general.



\---



\# Resultados principales



\## ROC-AUC en test



\- Regresión Logística: \~0.58

\- Random Forest: \~0.63

\- XGBoost: \~0.66



\## Validación cruzada optimizada



El modelo XGBoost optimizado alcanzó aproximadamente:



```text

ROC-AUC ≈ 0.978

```



durante validación cruzada sobre datos balanceados mediante SMOTE.



\---



\# Monitoreo de Data Drift



Se implementó monitoreo utilizando KS Test (Kolmogorov-Smirnov Test) para detectar cambios estadísticos entre datos históricos y datos actuales simulados.



Actualmente no se detectó drift significativo.



\---



\# Dashboard Streamlit



El proyecto incluye una aplicación interactiva desarrollada con Streamlit que permite visualizar:



\- Información general del dataset

\- Métricas del modelo

\- Estado de monitoreo

\- Variables más importantes



\---



\# Instalación



\## Clonar repositorio



```bash

git clone <URL\_DEL\_REPOSITORIO>

```



\## Crear entorno virtual



```bash

python -m venv venv

```



\## Activar entorno virtual



\### Windows



```bash

venv\Scripts\activate

```



\## Instalar dependencias



```bash

pip install -r requirements.txt

```



\---



\# Ejecución de Streamlit



```bash

streamlit run app/streamlit_app.py

```



\---

---

# API con FastAPI

El proyecto incluye una API desarrollada con FastAPI para exponer el modelo entrenado y permitir predicciones en tiempo real.

## Levantar la API localmente

```bash
uvicorn src.model_deploy:app --reload

Documentación automática

Una vez levantada la API, se puede acceder a la documentación Swagger desde:

http://127.0.0.1:8000/docs
Endpoint principal
POST /predict

Este endpoint recibe datos de entrada de un cliente y devuelve:

Predicción del modelo
Resultado interpretado
Probabilidad estimada de pago a tiempo

Ejemplo de respuesta:

{
  "prediccion": 0,
  "resultado": "Riesgo de no pago",
  "probabilidad_pago_a_tiempo": 0.2235
}

\# Autor



Yael Authier

