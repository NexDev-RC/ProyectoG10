"""
Entrenamiento con tracking de MLflow (Sprint 4).

Ejecuta el TrainPipeline completo registrando params, métricas y artefactos
en MLflow. El backend de tracking apunta a `paths.mlflow_uri` (mlruns/).

Uso:
    python scripts/run_mlflow_training.py            # sin Optuna (rápido)
    python scripts/run_mlflow_training.py --tune     # con Optuna

Luego inspeccionar los runs:
    mlflow ui                       # local, http://localhost:5000
    docker compose up mlflow        # contenedor, http://localhost:5000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.train_pipeline import TrainPipeline


def main():
    tune = "--tune" in sys.argv

    pipe = TrainPipeline()
    pipe.run(tune=tune, track_mlflow=True)

    print("\n=== Resumen ===")
    print("Features seleccionadas:", len(pipe.selected_features))
    print("Métricas (final_model):", pipe.metrics_report.get("final_model"))
    print("\nRun registrado en MLflow. Inspecciona con:  mlflow ui")


if __name__ == "__main__":
    main()
