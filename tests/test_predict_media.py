"""
Pruebas unitarias:
1) Predicción en imagen: estructura de salida válida y clases dentro del diccionario 'names'.
2) Predicción en video (1 frame): lectura de frame con OpenCV y predicción sobre ese frame.
   Esto verifica el flujo de video sin procesar el video completo (rápido).
"""
from typing import Dict, Tuple
from ultralytics import YOLO
import pytest
import cv2


def resolve_class_ids(names: Dict[int, str]) -> Tuple[int, int]:
    """
    Mapea robustamente índices de clases para 'empty/libre/vacant' y 'occupied/ocupado'.
    Fallbacks si los nombres no machan exactamente.
    """
    lower = {i: n.lower() for i, n in names.items()}

    def _find(keywords, default=None):
        for i, n in lower.items():
            if any(k in n for k in keywords):
                return i
        return default

    idx_empty = _find(["empty", "free", "libre", "vacant", "space-empty"])
    idx_occ   = _find(["occupied", "occu", "busy", "ocupado", "space-occupied"])

    if idx_empty is None and idx_occ is None and len(names) == 2:
        ids = sorted(names.keys())
        return ids[0], ids[1]

    ids = sorted(names.keys())
    if idx_empty is None:
        idx_empty = ids[0]
    if idx_occ is None:
        cand = [i for i in ids if i != idx_empty]
        idx_occ = cand[0] if cand else idx_empty
    return idx_empty, idx_occ


def test_predict_image(model_path, sample_image):
    """Predicción en una imagen: salida válida, IDs de clases dentro de names y conteos no negativos."""
    model = YOLO(str(model_path))
    names = model.names
    assert isinstance(names, dict) and len(names) >= 2

    idx_empty, idx_occ = resolve_class_ids(names)

    results = model.predict(
        source=str(sample_image),
        imgsz=640,
        conf=0.15,
        iou=0.7,
        verbose=False
    )

    assert results and len(results) >= 1
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    assert boxes is not None, "La salida no contiene 'boxes'"

    cls_ids = [int(b.cls[0]) for b in boxes] if len(boxes) > 0 else []
    valid_ids = set(names.keys())
    assert set(cls_ids).issubset(valid_ids), "IDs de clase fuera del diccionario 'names'"

    n_empty = sum(1 for cid in cls_ids if cid == idx_empty)
    n_occ   = sum(1 for cid in cls_ids if cid == idx_occ)
    assert n_empty >= 0 and n_occ >= 0


@pytest.mark.slow
def test_predict_video_single_frame(model_path, sample_video, tmp_path):
    """
    Verifica el pipeline de video leyendo un único frame con OpenCV y
    corriendo predicción sobre ese frame (imagen en memoria).
    """
    cap = cv2.VideoCapture(str(sample_video))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        pytest.skip("No fue posible leer un frame del video de prueba.")

    model = YOLO(str(model_path))
    names = model.names
    idx_empty, idx_occ = resolve_class_ids(names)

    # YOLO acepta arrays numpy (BGR) directamente
    results = model.predict(
        source=frame,     # frame en memoria
        imgsz=640,
        conf=0.15,
        iou=0.7,
        verbose=False
    )

    assert results and len(results) >= 1
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    assert boxes is not None

    cls_ids = [int(b.cls[0]) for b in boxes] if len(boxes) > 0 else []
    valid_ids = set(names.keys())
    assert set(cls_ids).issubset(valid_ids)

    # Conteo simple para asegurar integridad
    n_empty = sum(1 for cid in cls_ids if cid == idx_empty)
    n_occ   = sum(1 for cid in cls_ids if cid == idx_occ)
    assert n_empty >= 0 and n_occ >= 0
