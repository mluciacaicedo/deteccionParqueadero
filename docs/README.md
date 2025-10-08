
# Modelo utilizado
Modelo entrenado con YOLOv8 con Ultralytics + Tracking con MLflow.
- Modelo base por defecto: models/yolov8n.pt
- Dataset por defecto: data/mini_PKLot.v2-640.yolov8/data.yaml
- Modelo utilizado para la app: Modelo_yolov8_pklot2.pt
- MLflow Tracking URI: por defecto 'file://<RUTA_ABSOLUTA_DEL_PROYECTO>/mlruns'

Variables de entorno:
  MODEL_PATH=models/yolov8n.pt
  DATA_YAML=data/mini_PKLot.v2-640.yolov8/data.yaml
  EPOCHS=10
  BATCH=8
  IMG_SIZE=640
  EXP_NAME=train_pklot
  MLFLOW_TRACKING_URI=file:///...../mlruns


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
 Nuestra app de Detector_parqueaderos.py, se ejecuta a traves de Streamlit ubicada en la ruta src/streamlit_ui/app.py
Para el uso de la App: ejecuta el comando: `python Detector_parqueaderos.py`


# Pruebas Unitarias
  - Resolución robusta de IDs para 'libre' y 'ocupado'
  - Predicción sobre imagen con conteo por clase
Ejecuta las pruebas con:  `python -m pytest -vv -s`


