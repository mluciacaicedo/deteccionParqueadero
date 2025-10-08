"""
Lanza la interfaz de MLflow UI apuntando al directorio de runs (/mlruns).

Uso:
  python start_mlflow_ui.py
  python start_mlflow_ui.py --backend-uri file:///D:/Proyecto_FINAL_Repositorio/mlruns --port 5000 --host 127.0.0.1
  # O con entry point:
  start-mlflow-ui --port 5000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import webbrowser
from pathlib import Path


def _default_backend_uri() -> str:
    """
    Construye un backend-store-uri por defecto tipo file:///<abs>/mlruns
    si no está definido MLFLOW_TRACKING_URI.
    """
    # raíz del repo (este archivo suele vivir en la raíz)
    repo_root = Path(__file__).resolve().parent
    mlruns_path = (repo_root / "mlruns").resolve()
    return f"file:///{str(mlruns_path).replace(os.sep, '/')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanza MLflow UI.")
    parser.add_argument(
        "--backend-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI") or _default_backend_uri(),
        help="URI del backend store de MLflow (ej. file:///ruta/absoluta/mlruns).",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Puerto para MLflow UI (por defecto 5000)."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host para MLflow UI (por defecto 127.0.0.1)."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abrir automáticamente el navegador.",
    )
    args = parser.parse_args()

    # Refleja el tracking URI también en el entorno para coherencia con Ultralytics/MLflow
    os.environ["MLFLOW_TRACKING_URI"] = args.backend_uri

    url = f"http://{args.host}:{args.port}"
    print(f"Iniciando MLflow UI con backend-store-uri: {args.backend_uri}")
    print(f"URL esperada: {url}")

    try:
        # Ejecuta: python -m mlflow ui --backend-store-uri ... --port ... --host ...
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                args.backend_uri,
                "--port",
                str(args.port),
                "--host",
                args.host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # breve espera para levantar el servidor
        time.sleep(3)

        if not args.no-browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        print(f"MLflow UI lanzado en {url}")
        print("Presiona Ctrl+C para detenerlo.")
        process.wait()

    except KeyboardInterrupt:
        print("\n MLflow detenido manualmente.")
        try:
            process.terminate()
        except Exception:
            pass
    except FileNotFoundError:
        print("No se encontró 'mlflow'. Instálalo con: pip install mlflow")
    except Exception as e:
        print(f"Error al iniciar MLflow: {e}")


if __name__ == "__main__":
    main()