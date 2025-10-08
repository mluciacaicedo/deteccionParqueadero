# ---------------------------
# Etapa base: runtime Python
# ---------------------------
FROM python:3.12-slim

# Evitar prompts interactivos y logs truncados
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Paquetes del sistema necesarios para OpenCV y video
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copia solo requirements primero para maximizar capa de caché
COPY requirements.txt /app/requirements.txt

# Instala dependencias de Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia el resto del proyecto
COPY . /app

# Variables por defecto (puedes sobreescribirlas con -e en docker run)
# Cambia la ruta si cambias de modelo
ENV MODEL_PATH="models/Modelo_yolov8_pklot2.pt" \ 
    MLFLOW_TRACKING_URI="file:///app/mlruns"

# Streamlit: correr en 0.0.0.0 y puerto 8501 dentro del contenedor
EXPOSE 8501

# Evita que Streamlit necesite carpeta de config local
# y arranca la app directamente
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
