# MLflow – Versionado, monitoreo y retraining

Sprint 4 · Olist Revenue Forecast

## ¿Por qué este proyecto usa MLflow?

El forecast de ingresos se **ejecuta mensualmente con datos actualizados**. Cada
mes implica un nuevo entrenamiento (o actualización incremental) cuyo rendimiento
debe compararse contra el modelo en producción antes de promoverlo. MLflow cubre
exactamente esa necesidad:

| Necesidad del proyecto | Cómo lo resuelve MLflow |
|---|---|
| Comparar el modelo del mes N vs N-1 | Cada run guarda RMSE/MAPE/MAE → comparación en la UI |
| Versionar artefactos (`.pkl`) | `log_artifact` adjunta modelo y pipeline de limpieza al run |
| Trazabilidad / gobernanza | Tags (`version`, `model_type`) + lista de features por run |
| Reproducibilidad | Hiperparámetros (`log_params`) quedan ligados a cada run |
| Selección de campeón | Filtrar/ordenar runs por MAPE en la UI |

> **Qué proyectos del curso aplican a MLflow:** los que reentrenan periódicamente,
> comparan candidatos o necesitan versionado de artefactos. Este proyecto cumple
> los tres criterios, por lo que MLflow es plenamente justificable.

## Qué se registra en cada run

Implementado en `TrainPipeline._log_mlflow()` (`src/pipeline/train_pipeline.py`):

- **Tags:** `project`, `version`, `model_type`, `granularity`.
- **Params:** hiperparámetros de LightGBM (`best_params`) + `n_selected_features`.
- **Métricas técnicas:** `rmse`, `mape`, `mae`, `smape` (backtest).
- **Métricas de negocio:** prefijo `biz_*` (revenue, satisfacción, entrega…).
- **Artefactos:** `selected_features.txt`, `models/lgbm_forecaster.pkl`,
  `models/cleaning_pipeline.pkl`.

El backend de tracking apunta a `paths.mlflow_uri` del `config.yaml` (`mlruns/`).

> **Nota MLflow 3.x:** desde MLflow 3 el backend basado en carpeta (`mlruns/`)
> está en *maintenance mode* y el servidor/cliente lanza una excepción salvo que
> se active `MLFLOW_ALLOW_FILE_STORE=true`. El proyecto ya lo activa
> automáticamente: en el contenedor (variable de entorno en `docker-compose.yml`)
> y en el cliente (`TrainPipeline.run` hace `os.environ.setdefault`). Si en el
> futuro se quiere un backend de base de datos, usar
> `--backend-store-uri sqlite:///mlflow.db`.

## Cómo ejecutarlo

```bash
# Entrenamiento con tracking (rápido, sin Optuna)
python scripts/run_mlflow_training.py

# Con búsqueda de hiperparámetros
python scripts/run_mlflow_training.py --tune

# Inspeccionar los runs (local)
mlflow ui                       # http://localhost:5000

# Inspeccionar los runs (contenedor)
docker compose up mlflow        # http://localhost:5000
```

También se puede activar desde código:

```python
from src.pipeline.train_pipeline import TrainPipeline
TrainPipeline().run(tune=False, track_mlflow=True)
```

## Flujo de retraining mensual

```
        ┌─────────────────────────────────────────────────────────┐
        │  Llega el mes N (nuevos CSVs de Olist)                   │
        └─────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                        ▼
  (A) Actualización incremental            (B) Reentrenamiento completo
  PredictPipeline.run(                     scripts/run_mlflow_training.py
      new_month_data=df_mes_N)                 (--tune cada trimestre)
  → warm-start (forecaster.update)         → nuevo run MLflow con métricas
  → rápido, para forecast del mes            y artefactos versionados
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  Comparar en MLflow UI: MAPE_N vs MAPE_(N-1)             │
        │  ¿MAPE < 10% (target) y no degrada vs campeón?           │
        └─────────────────────────────────────────────────────────┘
              │ sí                                  │ no
              ▼                                      ▼
   Promover modelo a producción          Mantener campeón anterior
   (copiar .pkl a data/models/)          e investigar drift (PSI)
```

- **(A) Incremental:** barato, cada mes, para el forecast operativo
  (`forecaster.update()` añade árboles vía warm-start).
- **(B) Completo:** periódico (p. ej. trimestral) o cuando el drift lo amerite;
  re-ejecuta selección de features + Optuna y deja un run MLflow comparable.
- **Criterio de promoción (gobernanza):** MAPE de backtest < 10% y sin degradación
  frente al campeón vigente. Ver `docs/ETHICS_GOVERNANCE.md`.
- **Detección de drift:** el PSI (umbral 0.25) ya implementado en
  `src/features/selection.py` señala cuándo conviene forzar un reentrenamiento (B).
