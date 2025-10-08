"""
Entrenamiento YOLOv8 con Ultralytics + MLflow.

- Modelo base por defecto: models/yolov8n.pt
- Dataset por defecto: data/mini_PKLot.v2-640.yolov8/data.yaml
- MLflow Tracking URI: por defecto 'file://<RUTA_ABSOLUTA_DEL_PROYECTO>/mlruns'

Variables de entorno útiles:
  MODEL_PATH=models/yolov8n.pt
  DATA_YAML=data/mini_PKLot.v2-640.yolov8/data.yaml
  EPOCHS=10
  BATCH=8
  IMG_SIZE=640
  EXP_NAME=train_pklot
  MLFLOW_TRACKING_URI=file:///D:/Proyecto_FINAL_Repositorio/mlruns
"""

import os
import sys
from ultralytics import YOLO

try:
    import mlflow  #para mantener tracking
except Exception:
    mlflow = None


def _default_mlflow_uri() -> str:
    """Construye un file:// absoluto y válido para MLflow en Windows/Linux."""
    root = os.path.abspath(os.getcwd())
    mlruns_abs = os.path.join(root, "mlruns")
    os.makedirs(mlruns_abs, exist_ok=True)
    # Normaliza a file:///ruta/estilo/unix
    return "file:///" + mlruns_abs.replace("\\", "/")


def main() -> None:
    """Punto de entrada principal: configura MLflow, carga YOLO y entrena."""
    # ---------------- Configuración por variables de entorno ----------------
    model_path = os.getenv("MODEL_PATH", "models/yolov8n.pt")
    data_yaml = os.getenv("DATA_YAML", "data/mini_PKLot.v2-640.yolov8/data.yaml")

    epochs = int(os.getenv("EPOCHS", "10"))
    batch = int(os.getenv("BATCH", "8"))
    img_size = int(os.getenv("IMG_SIZE", "640"))
    exp_name = os.getenv("EXP_NAME", "train_pklot")

    # ------------------------- MLflow (opcional) ---------------------------
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", _default_mlflow_uri())
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri  # Ultralytics lo lee internamente

    # --------------------------- Mensajes útiles ---------------------------
    print("\n=== Configuración de entrenamiento ===")
    print(f"Modelo:        {model_path}")
    print(f"Dataset YAML:  {data_yaml}")
    print(f"Épocas:        {epochs}")
    print(f"Batch:         {batch}")
    print(f"Img size:      {img_size}")
    print(f"Experimento:   {exp_name}")
    print(f"MLflow URI:    {mlflow_uri}")
    print("======================================\n")

    # ------------------------- Validaciones rápidas ------------------------
    if not os.path.exists(model_path):
        sys.exit(f"No se encontró el modelo base: {model_path}")

    if not os.path.exists(data_yaml):
        sys.exit(f"No se encontró el archivo YAML del dataset: {data_yaml}")

    # ----------------------------- Entrenamiento ---------------------------
    model = YOLO(model_path)

    # Nota: Ultralytics detecta MLflow y loggea automáticamente si está disponible.
    # Si alguna vez quieres desactivar MLflow sin tocar el código:
    #   yolo settings mlflow=False
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name=exp_name,     # nombre del experimento/carpeta en runs/
        deterministic=True # resultados reproducibles (hasta donde permite CPU)
    )

    print("\n Entrenamiento finalizado. Modelos en: C:\\Users\\<tu_usuario>\\runs\\detect\\<experimento>\n"
          "   - weights/best.pt (mejor)\n"
          "   - weights/last.pt (última época)")


if __name__ == "__main__":
    main()
