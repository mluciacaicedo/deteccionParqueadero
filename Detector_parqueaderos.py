"""
Detector_parqueaderos.py
Este script ejecuta la app de Streamlit ubicada en la ruta src/streamlit_ui/app.py
Lanza `python -m streamlit run ...` en el puerto 8501
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "src" / "streamlit_ui" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"No se encontró la app de Streamlit en: {app_path}")

    # Modelo por defecto si el usuario no define MODEL_PATH
    default_model = (repo_root / "models" / "Modelo_yolov8_pklot2.pt").resolve()

    env = os.environ.copy()
    env.setdefault("MODEL_PATH", str(default_model))

    # Puedes cambiar puerto/headless si quieres
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port=8501",
        "--server.headless=true",
    ]

    print(f"Lanzando Streamlit: {' '.join(cmd)}")
    print(f"MODEL_PATH = {env['MODEL_PATH']}")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
