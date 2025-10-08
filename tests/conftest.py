"""
Fixtures compartidas para pruebas de inferencia con YOLOv8.
- Modelo por defecto: models/Modelo_yolov8_pklot2.pt
- Permite override con la variable de entorno MODEL_PATH.
- Busca una imagen y un video de ejemplo o usa TEST_IMAGE / TEST_VIDEO.
"""
import os
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def sample_image(repo_root: Path) -> Path:
    
    img = repo_root / "bus.jpg"
    if not img.exists():
        pytest.skip("No hay imagen de prueba 'bus.jpg'")
    return img

@pytest.fixture(scope="session")
def model_path(repo_root: Path) -> Path:
    # Usa tu modelo entrenado
    candidates = [
        Path(os.getenv("MODEL_PATH", "")),
        repo_root / "models" / "Modelo_yolov8_pklot2.pt",
        repo_root / "models" / "yolov8n.pt",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    pytest.skip("No hay pesos .pt disponibles para pruebas lentas")
