# =================================================
# Imagen base
# =================================================
# Se utiliza una imagen liviana de Python 3.11
FROM python:3.11-slim


# =================================================
# Directorio de trabajo
# =================================================
# Define la carpeta interna donde se ejecutará la aplicación
WORKDIR /app


# =================================================
# Instalación de dependencias
# =================================================
# Primero se copia requirements.txt para instalar las librerías necesarias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# =================================================
# Copia del código fuente
# =================================================
# Copia todos los archivos del proyecto dentro del contenedor
COPY . .


# =================================================
# Exposición del puerto
# =================================================
# FastAPI se ejecutará en el puerto 8000
EXPOSE 8000


# =================================================
# Comando de ejecución
# =================================================
# Levanta la API utilizando Uvicorn
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]