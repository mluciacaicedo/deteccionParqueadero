"""
Fixtures y utilidades compartidas para las pruebas.
Ejecuta las pruebas con:  python -m pytest -vv -s
"""

from pathlib import Path
import os
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "models" / "Modelo_yolov8_pklot2.pt"
DEFAULT_IMAGE = REPO /  ".\data\external\parqueadero_test.jpg"  # seleccionar imagen

@pytest.fixture(scope="session")
def model_path() -> Path:
    p = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
    if not p.exists():
        pytest.skip(f"Modelo no encontrado: {p}")
    print(f"[SETUP] Modelo: {p}")
    return p

@pytest.fixture(scope="session")
def sample_image() -> Path:
    img = Path(os.getenv("TEST_IMAGE", DEFAULT_IMAGE))
    if not img.exists():
        pytest.skip(f"Imagen de prueba no encontrada: {img}")
    print(f"[SETUP] Imagen: {img}")
    return img
