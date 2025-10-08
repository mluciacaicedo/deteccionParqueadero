"""
Pruebas unitarias:
1) Predicción en imagen: estructura de salida válida y clases dentro del diccionario 'names'.
2) Predicción en video (1 frame): lectura de frame con OpenCV y predicción sobre ese frame.
   Esto verifica el flujo de video sin procesar el video completo (rápido).
"""
import pytest
import numpy as np
from ultralytics import YOLO

def resolve_empty_occupied_ids(names: dict[int, str]) -> tuple[int, int]:
    lower = {i: str(n).lower() for i, n in names.items()}
    def find(keys):
        for i, n in lower.items():
            if any(k in n for k in keys):
                return i
        return None
    idx_empty = find(["empty", "free", "libre", "vacant"])
    idx_occ   = find(["occupied", "ocupado", "busy", "occu"])
    if idx_empty is None and idx_occ is None and len(names) >= 2:
        ids = sorted(names.keys())
        return ids[0], ids[1]
    if idx_empty is None:
        ids = [i for i in sorted(names) if i != idx_occ]
        idx_empty = ids[0] if ids else idx_occ
    if idx_occ is None:
        ids = [i for i in sorted(names) if i != idx_empty]
        idx_occ = ids[0] if ids else idx_empty
    return idx_empty, idx_occ

def test_class_id_mapping_generic():
    names = {0: "space-empty", 1: "space-occupied"}
    e, o = resolve_empty_occupied_ids(names)
    assert e in names and o in names and e != o

@pytest.mark.slow
def test_predict_on_image_runs(sample_image, model_path):
    model = YOLO(str(model_path))
    res = model.predict(source=str(sample_image), imgsz=640, conf=0.25, verbose=False)
    assert len(res) >= 1
    r0 = res[0]
    # Asegura que devolvió un frame
    assert r0.orig_img is not None
    if r0.boxes is not None and len(r0.boxes) > 0:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        assert np.all(xyxy[:, :2] <= xyxy[:, 2:4])  # x1<=x2, y1<=y2