# Imagen base un poco más completa
FROM python:3.10-bullseye

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias primero
COPY requirements.txt .

# Instalar dependencias del sistema necesarias para torch, opencv, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto para Streamlit
EXPOSE 8501

# Comando por defecto
CMD ["streamlit","run","app.py","--server.address","0.0.0.0","--server.port","8501","--server.headless","true"]
