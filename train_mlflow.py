"""
Train a model and prepare metadata for logging

1) Carga config de dataset y pesos base YOLOv8.
2) Entrena un modelo:
  MODEL_PATH=models/yolov8n.pt
  DATA_YAML=data/mini_PKLot.v2-640.yolov8/data.yaml
  EPOCHS=10
  BATCH=8
  IMG_SIZE=640
  EXP_NAME=train_pklot
3) Evalúa (val) el modelo resultante.
4) Prepara hiperparámetros y métricas.
5) Registra todo en MLflow (params, metrics, artifacts como best.pt).

"""

import os
from pathlib import Path
from ultralytics import YOLO
import mlflow

# ----------------- Configs rápidas -----------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
DATA_YAML  = os.getenv("DATA_YAML", "data/mini_PKLot.v2-640.yolov8/data.yaml")
EPOCHS     = int(os.getenv("EPOCHS", "10"))
BATCH      = int(os.getenv("BATCH", "8"))
IMG_SIZE   = int(os.getenv("IMG_SIZE", "640"))
EXP_NAME   = os.getenv("EXP_NAME", "train_pklot")
DEVICE     = os.getenv("DEVICE", "cpu")  # 'cpu' o 'cuda:0'


# Set our tracking server uri for logging

# URI de tracking (si no defines HTTP, caerá a file://./mlruns)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                                  "file:///" + (Path.cwd()/"mlruns").as_posix()))

#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080") #Para usarlo debemos tener LANZADO el SERVER (mlflow server --host 127.0.0.1 --port 8080)

# Create a new MLflow Experiment
mlflow.set_experiment("Detector_parqueaderos_UAO-YOLOv8")

# Entrenamiento YOLOv8
model = YOLO(MODEL_PATH)
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    name=EXP_NAME,
    device=DEVICE,
    deterministic=True
)

# ----------------- MLflow logging (estilo guía) -----------------
params = {
    "model_path": MODEL_PATH,
    "data_yaml": DATA_YAML,
    "epochs": EPOCHS,
    "batch": BATCH,
    "imgsz": IMG_SIZE,
    "exp_name": EXP_NAME,
    "device": DEVICE,
}

with mlflow.start_run():  # Start an MLflow run
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss/metrics
    mets = getattr(results, "metrics", {}) or {}
    
    # Log the loss metric(s) – adaptado a YOLOv8
    for k in ["precision", "recall", "mAP50", "mAP50-95"]:
        if k in mets:
            mlflow.log_metric(k, float(mets[k]))

    # Infer the model signature (para visión lo dejamos None)
    signature = None

    # Log the model (PyTorch)
    # intenta loguear el nn.Module interno de Ultralytics
    logged_ok = False
    try:
        import mlflow.pytorch as mlpt
        mlpt.log_model(
            pytorch_model=model.model,      # nn.Module interno
            artifact_path="model",          # carpeta dentro del run
            registered_model_name="parking-yolov8",  # opcional: registra en Model Registry
        )
        logged_ok = True
    except Exception as e:
        print(f"[MLflow] No se pudo loguear nn.Module con mlflow.pytorch: {e}")

    # Log the model (pesos .pt) – opción B (fallback):
    save_dir = Path(getattr(results, "save_dir", "") or "")
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if best_path.exists():
        mlflow.log_artifact(str(best_path), artifact_path="weights")
        logged_ok = True
    if last_path.exists():
        mlflow.log_artifact(str(last_path), artifact_path="weights")

    # Plots típicos de YOLO si existen
    for fname in ["results.png", "confusion_matrix.png", "labels.jpg", "val_batch0_pred.jpg"]:
        f = save_dir / fname
        if f.exists():
            mlflow.log_artifact(str(f), artifact_path="plots")

    # Set a tag (context info)
    mlflow.set_tags({
        "project": "Detector_parqueaderos",
        "framework": "ultralytics-yolov8",
        "note": "Entrenamiento del modelo con Mlflow",
    })

print("✅ Listo: ejecuta 'mlflow ui' (o tu servidor) para ver el run.")