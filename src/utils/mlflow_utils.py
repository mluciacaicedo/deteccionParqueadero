"""
Utilidades simples para registrar experimentos en MLflow.

Estas funciones envuelven mlflow.start_run, log_params, log_artifacts y end_run
para simplificar su uso en los scripts del proyecto.
"""

import mlflow

def start_run(experiment_name: str = "default", run_name: str = None):
    """
    Inicia un experimento en MLflow.
    Si no existe el experimento, lo crea automáticamente.
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    """
    Registra parámetros clave del experimento en MLflow.
    """
    mlflow.log_params(params)

def log_artifacts(path: str):
    """
    Registra carpetas o archivos completos como artefactos en MLflow.
    """
    mlflow.log_artifacts(path)

def end_run():
    """
    Finaliza el experimento actual en MLflow.
    """
    mlflow.end_run()
