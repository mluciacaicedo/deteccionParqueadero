"""
Fixtures compartidas para pruebas de inferencia con YOLOv8.
- Modelo por defecto: models/Modelo_yolov8_pklot2.pt
- Permite override con la variable de entorno MODEL_PATH.
- Busca una imagen y un video de ejemplo o usa TEST_IMAGE / TEST_VIDEO.
"""
from pathlib import Path
import os
import pytest


def _first_match(glob_patterns, exts=None):
    """Devuelve el primer Path que cumpla algún patrón y extensión (si se indica)."""
    for pat in glob_patterns:
        for p in Path(".").glob(pat):
            if p.is_file() and (exts is None or p.suffix.lower() in exts):
                return p
    return None


@pytest.fixture(scope="session")
def model_path():
    """
    Localiza el modelo a usar en pruebas.
    Orden de prioridad:
      1) $MODEL_PATH
      2) models/Modelo_yolov8_pklot2.pt
      3) runs previos (best.pt)
    """
    candidates = [
        os.getenv("MODEL_PATH"),
        "models/Modelo_yolov8_pklot2.pt",
        "models/best.pt",
        "runs/detect/train_pklot2/weights/best.pt",
        "runs/detect/train18/weights/best.pt",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip("No se encontró un modelo válido. Define MODEL_PATH o coloca tu best.pt en models/.")


@pytest.fixture(scope="session")
def sample_image():
    """
    Imagen de prueba:
      - Usa $TEST_IMAGE si existe.
      - Si no, busca en data/**/images/(val|train)/ la primera imagen válida.
    """
    env_img = os.getenv("TEST_IMAGE")
    if env_img and Path(env_img).is_file():
        return Path(env_img)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    patterns = [
        "data/**/images/val/*.*",
        "data/**/images/train/*.*",
        "samples/*.*",
    ]
    p = _first_match(patterns, img_exts)
    if p:
        return p

    pytest.skip("No se encontró una imagen de prueba. Define TEST_IMAGE o coloca imágenes en data/**/images/(val|train)/.")


@pytest.fixture(scope="session")
def sample_video():
    """
    Video de prueba:
      - Usa $TEST_VIDEO si existe.
      - Si no, busca un archivo de video en data/**/videos o samples/.
    Si no hay video, se marca como SKIP (las pruebas de video se saltarán).
    """
    env_vid = os.getenv("TEST_VIDEO")
    if env_vid and Path(env_vid).is_file():
        return Path(env_vid)

    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    patterns = [
        "data/**/videos/*.*",
        "samples/*.*",
        "data/**/*.mp4",
    ]
    p = _first_match(patterns, vid_exts)
    if p:
        return p

    pytest.skip("No se encontró un video de prueba. Define TEST_VIDEO o coloca un video en data/**/videos/ o samples/.")
