"""
Streamlit UI para detección de parqueaderos con YOLOv8.

- Carga un modelo .pt desde env var MODEL_PATH o por defecto: /models/Modelo_yolov8_pklot2.pt
- Soporta imagen y video (procesamiento frame a frame con OpenCV).
- Mapeo robusto de clases: 'empty|free|libre|vacant' y 'occupied|ocupado|busy' (o fallbacks por índice).
- Parámetros ajustables: conf, iou, imgsz.
"""

from __future__ import annotations

import os
import io
import time
import tempfile
from pathlib import Path
import tempfile
from typing import Tuple, Dict

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ------------------------------------------------------------
# Rutas y carga de modelo (independiente del directorio actual)
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Proyecto_FINAL_Repositorio
DEFAULT_MODEL = (REPO_ROOT / "models" / "Modelo_yolov8_pklot2.pt").resolve()

MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL))


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """Carga y cachea el modelo YOLO."""
    return YOLO(model_path)


# Cargar modelo con spinner
with st.spinner(f"Cargando modelo: {Path(MODEL_PATH).name}"):
    model = load_model(MODEL_PATH)

# ------------------------------------------------------------
# Utilidades de clases y conteo
# ------------------------------------------------------------
NAMES: Dict[int, str] = model.names if hasattr(model, "names") else {}
LOWER_NAMES = {i: n.lower() for i, n in NAMES.items()}


def _find_class_id(keywords, default=None):
    for i, n in LOWER_NAMES.items():
        if any(k in n for k in keywords):
            return i
    return default


# Detectar ids para libre/ocupado
IDX_EMPTY = _find_class_id(["empty", "free", "libre", "vacant"], default=None)
IDX_OCC = _find_class_id(["occupied", "occu", "busy", "ocupado"], default=None)

# Fallbacks si nombres no coinciden
if IDX_EMPTY is None and IDX_OCC is None and len(NAMES) == 2:
    IDX_EMPTY, IDX_OCC = 0, 1
else:
    ids_sorted = sorted(list(NAMES.keys())) if NAMES else [0, 1]
    if IDX_EMPTY is None:
        IDX_EMPTY = ids_sorted[0]
    if IDX_OCC is None:
        IDX_OCC = ids_sorted[1] if len(ids_sorted) > 1 else ids_sorted[0]

COLOR_EMPTY = (0, 200, 0)   # BGR (verde)
COLOR_OCC = (0, 0, 200)     # BGR (rojo)
COLOR_OTHER = (200, 160, 0) # BGR (azul-amarillo)


def draw_boxes_and_count(img_bgr: np.ndarray,
                         results,
                         idx_empty: int,
                         idx_occ: int) -> Tuple[np.ndarray, int, int]:
    """
    Dibuja cajas y contabiliza por clase. Devuelve imagen anotada y conteos (libres, ocupados).
    """
    libres = 0
    ocupados = 0
    out = img_bgr.copy()

    if not results or len(results) == 0:
        return out, libres, ocupados

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out, libres, ocupados

    for box in r0.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == idx_empty:
            color = COLOR_EMPTY
            libres += 1
        elif cls == idx_occ:
            color = COLOR_OCC
            ocupados += 1
        else:
            color = COLOR_OTHER

        label = f"{NAMES.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Cabecera con totales
    header = f"libres={libres} | ocupados={ocupados}"
    cv2.rectangle(out, (0, 0), (max(180, out.shape[1] // 4), 26), (32, 32, 32), -1)
    cv2.putText(out, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out, libres, ocupados


def predict_image(img_bgr: np.ndarray, conf: float, iou: float, imgsz: int):
    """Corre inferencia YOLO sobre una imagen BGR y devuelve results."""
    return model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )


def process_video(input_path: str, output_path: str, conf: float, iou: float, imgsz: int) -> Dict[str, float]:
    """
    Procesa un video frame a frame y guarda el video anotado.
    Devuelve métricas simples (fps aproximado y promedios de conteo).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video de entrada.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Limitar FPS de procesamiento
    target_fps = 5 #definir mas o menos cuadros por segundo
    frame_skip = int(fps_in // target_fps) if fps_in > target_fps else 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = st.progress(0, text="Procesando video...")
    t0 = time.time()

    sum_libre, sum_ocup, frames = 0, 0, 0

    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #Saltar cuadros según el frame_skip
            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            results = predict_image(frame, conf, iou, imgsz)
            annotated, libres, ocupados = draw_boxes_and_count(frame, results, IDX_EMPTY, IDX_OCC)
            out.write(annotated)

            sum_libre += libres
            sum_ocup += ocupados
            frames += 1
            frame_id += 1

            if total_frames > 0:
                pbar.progress(min(frames / total_frames, 1.0))

    finally:
        cap.release()
        out.release()
        pbar.empty()

    elapsed = time.time() - t0
    fps_proc = frames / elapsed if elapsed > 0 else 0.0

    return {
        "frames": frames,
        "fps_in": float(fps_in),
        "fps_proc": fps_proc,
        "avg_libres": (sum_libre / frames) if frames else 0.0,
        "avg_ocupados": (sum_ocup / frames) if frames else 0.0,
    }

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Detector de Parqueaderos (YOLOv8)", page_icon="🅿️", layout="wide")

st.title("🅿️ Detector de Parqueaderos")
#st.caption(f"Modelo: `{Path(MODEL_PATH).name}` | Clases: {NAMES}  | empty={IDX_EMPTY}, occupied={IDX_OCC}")

with st.sidebar:
    st.header("Parámetros")
    mode = st.radio("¿Qué deseas cargar?", ["Imagen", "Video"], index=0, horizontal=True)
    conf = st.slider("Nivel de confianza", 0.1, 0.9, 0.5, 0.05, help="Entre más alta, el modelo mostrará solo detecciones muy seguras.")
    iou = st.slider("Nivel de superposición (IoU)", 0.1, 0.9, 0.7, 0.05, help="Controla cuánta coincidencia se permite entre las detecciones.")
    #imgsz = st.select_slider("Calidad del procesamiento", options=[320, 416, 512, 640, 768], value=640, help="Tamaño de imagen que se usa para analizar. Mayor tamaño = más precisión, pero más lento.")
    imgsz = 640
if mode == "Imagen":
    st.subheader("📷 Subir imagen")
    up = st.file_uploader("Arrastra o selecciona una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if up is not None:
        # Leer bytes → np array BGR
        bytes_data = up.read()
        img_arr = np.frombuffer(bytes_data, np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen original", width="stretch")

        with st.spinner("Inferencia..."):
            results = predict_image(img_bgr, conf, iou, imgsz)
            annotated, libres, ocupados = draw_boxes_and_count(img_bgr, results, IDX_EMPTY, IDX_OCC)

        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detecciones", width="stretch")

        st.success(f"**Conteo** → Libres: `{libres}`  |  Ocupados: `{ocupados}`")

        # Descarga de imagen anotada
        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            st.download_button(
                "⬇️ Descargar imagen anotada",
                data=buf.tobytes(),
                file_name="predicted.jpg",
                mime="image/jpeg",
            )

else:
    st.subheader("🎬 Subir video")
    upv = st.file_uploader("Arrastra o selecciona un video (MP4/MOV/AVI)", type=["mp4", "mov", "avi", "mkv"])
    if upv is not None:
        # Guardar a archivo temporal de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upv.name).suffix) as tmp_in:
            tmp_in.write(upv.read())
            tmp_in.flush()
            in_path = tmp_in.name

        # Archivo temporal de salida
        out_path = Path(tempfile.gettempdir()) / f"predicted_{Path(upv.name).stem}.mp4"

        with st.spinner("Procesando video…"):
            stats = process_video(str(in_path), str(out_path), conf, iou, imgsz)

        st.video(str(out_path))
        st.success(
            f"**Frames**: {stats['frames']} | "
            f"**FPS in**: {stats['fps_in']:.1f} | **FPS proc**: {stats['fps_proc']:.1f} | "
            f"**Promedios** → libres: {stats['avg_libres']:.1f}, ocupados: {stats['avg_ocupados']:.1f}"
        )

        # Descargar video anotado
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Descargar video anotado",
                data=f.read(),
                file_name=f"predicted_{Path(upv.name).stem}.mp4",
                mime="video/mp4",
            )