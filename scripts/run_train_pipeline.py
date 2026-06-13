"""
Script de humo (smoke test) para validar el TrainPipeline (Sprint 1+2)
de extremo a extremo, sin tuning de hiperparámetros (tune=False).

Uso:
    python scripts/run_train_pipeline.py
"""
from src.pipeline.train_pipeline import TrainPipeline

pipe = TrainPipeline()

pipe.run(tune=False)

print('Features seleccionadas:', pipe.selected_features)

print('Métricas:', pipe.metrics_report.get('final_model'))
