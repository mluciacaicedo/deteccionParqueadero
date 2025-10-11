"""
Arranca la UI de MLflow apuntando al backend local (carpeta mlruns).
Uso: python start_mlflow_ui.py
port 5000 --host 127.0.0.1
Luego abre: http://127.0.0.1:5000

"""
import os, sys, subprocess

def main():
    mlruns_path = os.path.abspath("mlruns")
    uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
    host = "127.0.0.1"
    port = "5000"

    print(f"Iniciando MLflow UI en http://{host}:{port}")
    print(f"Backend store: {uri}")

    # usa el mismo intérprete que ejecuta este script
    py = sys.executable

    subprocess.run(
        [py, "-m", "mlflow", "ui",
         "--backend-store-uri", uri,
         "--host", host,
         "--port", port],
        check=True
    )

if __name__ == "__main__":
    main()
