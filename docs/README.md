
## Hola! Bienvenido a la herramienta para la detección Parqueaderos libres con YOLOv8

Aplicación basada en visión computacional para **detectar espacios libres y ocupados** en tiempo real, muestra que espacios de un parqueadero están libres u ocupados a partir de imágenes o video de cámaras ya instaladas. El sistema aprende a reconocer plazas de parqueo y muestra los resultados de forma clara, con conteos y resaltando cada espacio detectado. 

Incluye: entrenamiento con PKLot y CNRPark-EXT, **tracking** con MLflow, **servicio gRPC** de inferencia y empaquetado **Docker**.


Para facilitar su uso y evolución, el proyecto incluye una aplicación web sencilla **con Streamlit** para cargar imágenes o videos, un registro automático del desempeño de los entrenamientos, y un servicio independiente de inferencia que permite integrarlo con otras aplicaciones. Además, se puede empaquetar en Docker para desplegarlo de forma rápida y consistente en distintos entornos. En resumen, es una herramienta lista para pruebas en campo, pensada para reducir tiempos de búsqueda, congestión y costos operativos en la gestión de parqueaderos.


## Caracteristicas:

- Detección de **space-empty** / **space-occupied** (mapeo robusto de nombres).
- Incluye un **Subconjunto de datos balanceado** extraidos del Datset original, para pruebas rapidas.
- App web **Streamlit**: carga de **imagen** o **video**, conteo + overlays.
- Tracking con **MLflow**: hiperparámetros, métricas y gráficos.
- Empaquetado con **Docker**: imagen para demo web o servidor gRPC.
- Pruebas unitarias con **Pytest**: pruebas mínimas para predicción e identificación de clases.

---

## Estructura del repositorio

DeteccionParqueadero/
├─ models/
│  └─ Modelo_yolov8_pklot2.pt          # Modelo YOLOv8 por defecto (usado por la app)
├─ data/                                # Datos de ejemplo / insumos opcionales
├─ docs/                                # Documentación del proyecto, README, LICENCIAS
├─ mlartifacts/                         # Artefactos de entrenamiento/experimentación (opcional)
├─ mlruns/                              # Carpeta de MLflow (experimentos locales)
├─ protos/                              # (Reservado) contratos/protos si se usan en el futuro
├─ src/                                 # Código auxiliar (utilidades, scripts internos)
├─ tests/
│  ├─ conftest.py
│  └─ test_predict_media.py             # Pruebas (pytest)
├─ .dockerignore
├─ .gitignore
├─ .gitlab-ci.yml
├─ .python-version
├─ app.py                               # App Streamlit principal (imagen/video)
├─ Dockerfile                           # Imagen para desplegar la app
├─ Makefile                             # Atajos (make cod / make test)
├─ main.py                              # Scripts auxiliares (si aplica)
├─ predict_video.py                     # CLI para procesar video y guardar resultado
├─ pyproject.toml
├─ requirements.txt
├─ resultado_annotated.jpg              # Ejemplo de salida anotada
├─ start_mlflow_ui.py                   # Lanza MLflow UI local
└─ train_mlflow.py                      # Entrenamiento YOLOv8 + tracking en MLflow


## Instalación y ejecución:

# Instalación local
# 1. Clonar repositorio: 
git clone https://github.com/mluciacaicedo/deteccionParqueadero.git
cd deteccionParqueadero

# 2.Activar Entorno
python -m venv .venv
  
  # Si usas UV
  uv venv .venv

# Para Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Instalacion de Dependencias
pip install -r requirements.txt

# si usas UV
uv pip install -r requirements.txt


## Datasets y creación de subconjunto:

Coloca tu dataset YOLOv8 (export de Roboflow) en la carpeta `data/PKLot.v2-640.yolov8/`.

# Genera un subset balanceado 80/20 ejecutando:
python src/utils/create_subset.py --root data/PKLot.v2-640.yolov8 --out data/mini_PKLot.v2-640.yolov8

Nota: El script detecta clases automáticamente (space-empty / space-occupied) y crea data.yaml en la carpeta del subset.


## Entrenamiento del Modelo
Para ejecutar el entrenamiento, usa el comando `Python train.py` para entrenar el modelo YOLOv8

Al ejecutar, se carga el modelo base (yolov8n.pt el modelo más liviano de YOLOv8 para entrenamientos rápidos) para entrenar con el modelo con el subconjunto de PKLot. 

`train.py` usa por defecto:
- Modelo base: `models/yolov8n.pt`
- Dataset: `data/mini_PKLot.v2-640.yolov8/data.yaml`
- MLflow: `file://./mlruns`

Al terminar el entrenamiento  con un modelo con YOLOv8, el mejor modelo se guarda en un archivo con extensión: `runs/detect/<exp>/weights/best.pt`  “best.pt” es el modelo obtenido. Renombralo como `Modelo_yolov8_pklot2` y pegalo en la carpera models.

Durante el entrenamiento, el tracking se realiza con MLflow, el cual nos permite ver las métricas del entrenamiento, hiperparámetros, artefactos y cual fue el mejor modelo.

Visualiza las metricas del entrenamiento en la carpeta: `./mlruns`


# Pruebas de desempeño del Modelo

Este repositorio ofrece la posibilidad de realizar Inferencia de video con YOLOv8 (Ultralytics) para conteo de plazas de parqueo, sin el uso de la App para validar el rendimiento del Modelo.
- Acepta archivo de video o webcam.
- Dibuja boxes y muestra/guarda el video anotado.
- Cuenta por frame clases "libres" y "ocupadas" detectándolas por nombre

Uso:
Prueba con imagen: `python predict_video.py --source ".\data\external\parqueader_test.jpg" --model ".\models\Modelo_yolov8_pklot2.pt" --save out.mp4 –show`

Prueba con video: `python predict_video.py --source ".\data\external\video 1_1.mp4" --model ".\models\Modelo_yolov8_pklot2.pt" --save out.mp4 –show`

webcam: `python predict_video.py --source 0 --show`

Esto generará predicciones que se guardan como: out.mp4

# Ejecucion de la app:
 Nuestra app de Detector_parqueaderos, se ejecuta a traves de Streamlit, para su uso, 
 ejecuta el comando: `uv run streamlit run app.py` inmediatamente se te abrirá el navegador, y podras usar la aplicacion.

# Pruebas Unitarias
  - Resolución robusta de IDs para 'libre' y 'ocupado'
  - Predicción sobre imagen con conteo por clase
Ejecuta las pruebas con:  `python -m pytest -vv -s`


