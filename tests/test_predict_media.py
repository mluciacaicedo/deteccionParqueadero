"""
Pruebas verbosas:
  - Resolución robusta de IDs para 'libre' y 'ocupado'
  - Predicción sobre imagen con conteo por clase
Ver los prints con: python -m pytest -vv -s
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import pytest
from ultralytics import YOLO


# ----------------- helpers (mismos que usas en la app) -----------------
def _normalize_class_map(names: Dict[int, str]) -> Dict[int, str]:
    return {i: str(n).lower() for i, n in names.items()}

def _find_class_id(names_lower: Dict[int, str], keywords: Iterable[str]) -> Optional[int]:
    for i, n in names_lower.items():
        if any(k in n for k in keywords):
            return i
    return None

def resolve_empty_occupied_ids(names: Dict[int, str]) -> tuple[int, int]:
    lower = _normalize_class_map(names)
    idx_empty = _find_class_id(lower, ["empty", "free", "libre", "vacant"])
    idx_occ   = _find_class_id(lower, ["occupied", "ocupado", "busy", "occu"])

    if idx_empty is None and idx_occ is None:
        ids_sorted = sorted(list(names.keys()))
        if len(ids_sorted) >= 2:
            return ids_sorted[0], ids_sorted[1]
        elif len(ids_sorted) == 1:
            return ids_sorted[0], ids_sorted[0]
        return 0, 1

    if idx_empty is None:
        ids_sorted = [i for i in sorted(names) if i != idx_occ]
        idx_empty = ids_sorted[0] if ids_sorted else idx_occ
    if idx_occ is None:
        ids_sorted = [i for i in sorted(names) if i != idx_empty]
        idx_occ = ids_sorted[0] if ids_sorted else idx_empty
    return idx_empty, idx_occ
# ----------------------------------------------------------------------


def test_resolucion_ids_clase_verbose(model_path: Path):
    print("[STEP] Cargando modelo…")
    model = YOLO(str(model_path))

    print("[INFO] Clases del modelo:")
    for k, v in model.names.items():
        print(f"  id={k} -> '{v}'")

    idx_empty, idx_occ = resolve_empty_occupied_ids(model.names)
    print(f"[RESULT] idx_empty={idx_empty}  idx_occ={idx_occ}")

    assert isinstance(idx_empty, int)
    assert isinstance(idx_occ, int)
    assert idx_empty != idx_occ or len(model.names) == 1


@pytest.mark.timeout(60)
def test_predict_on_image_verbose(sample_image: Path, model_path: Path):
    print("[STEP] Leyendo imagen…")
    img = cv2.imread(str(sample_image))
    assert img is not None, f"No se pudo leer la imagen: {sample_image}"
    print(f"[INFO] Imagen shape={img.shape}")

    print("[STEP] Cargando modelo…")
    model = YOLO(str(model_path))

    print("[STEP] Resolviendo IDs…")
    idx_empty, idx_occ = resolve_empty_occupied_ids(model.names)
    print(f"[INFO] idx_empty={idx_empty}  idx_occ={idx_occ}")

    print("[STEP] Ejecutando predict…")
    res = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)[0]

    num = int(res.boxes.shape[0]) if res.boxes is not None else 0
    cls_ids = res.boxes.cls.int().tolist() if res.boxes is not None else []
    libres = sum(1 for cid in cls_ids if cid == idx_empty)
    ocup   = sum(1 for cid in cls_ids if cid == idx_occ)

    print(f"[RESULT] detecciones={num}  libres={libres}  ocupadas={ocup}")

    # “Smoke test”: solo verificamos que corre sin excepciones.
    assert num >= 0

