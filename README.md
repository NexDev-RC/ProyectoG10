# Olist Revenue Forecast

Forecast de **ingresos mensuales** (`monthly_revenue`) del e-commerce brasileño
Olist mediante series temporales (LightGBM, estrategia *direct multi-step*).
Métricas objetivo: **RMSE / MAPE** (target MAPE < 10 %).

Proyecto del Módulo 13 (Maestría) — Grupo 10. Cubre Sprints 1–4:
EDA → pipeline reproducible → modelo final → **integración, despliegue y gobernanza**.

---

## Estructura

```
api/         API REST (FastAPI)
dashboard/   Dashboard interactivo (Streamlit)
src/         Código fuente (data, features, models, evaluation, pipeline)
config/      config.yaml (paths, splits, hiperparámetros, supuestos de negocio)
data/        raw/ (CSVs Olist), processed/ (parquet), models/ (.pkl)
docs/        MLFLOW.md, ETHICS_GOVERNANCE.md
notebooks/   Notebooks por sprint
scripts/     Runners de entrenamiento
reports/     Gráficos y entregables
tests/       Pruebas con pytest
```

## Instalación

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

## Entrenamiento

```bash
# Pipeline completo (rápido, sin Optuna)
python scripts/run_train_pipeline.py

# Con tracking MLflow
python scripts/run_mlflow_training.py            # añade --tune para Optuna
```

Genera los artefactos en `data/models/` (`lgbm_forecaster.pkl`,
`cleaning_pipeline.pkl`) y `data/processed/monthly_features.parquet`.

## Uso – API

```bash
uvicorn api.main:app --reload --port 8000
# Documentación interactiva: http://localhost:8000/docs
```

| Endpoint | Descripción |
|---|---|
| `GET /api/v1/health` | Estado del servicio |
| `GET /api/v1/forecast?horizon=3` | Forecast de los próximos N meses |
| `GET /api/v1/metrics` | Métricas técnicas (backtest) |
| `POST /api/v1/predict` | Predicción con features custom |
| `GET /api/v1/feature-importance` | Importancia (splits LightGBM) |
| `GET /api/v1/shap?top_n=20&horizon=1` | **Importancia SHAP (Sprint 4)** |

## Uso – Dashboard

```bash
streamlit run dashboard/app.py     # http://localhost:8501
```

Secciones: Overview · Forecast · Features · **SHAP** · Métricas (con **ROI**) · What-If.

## Despliegue con Docker (Sprint 4)

```bash
docker compose up --build
```

| Servicio | URL |
|---|---|
| API REST | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

> Los servicios montan `data/models/` y `data/processed/` como volúmenes de solo
> lectura, por lo que los artefactos deben existir antes de levantar los contenedores.

## Gobernanza y MLOps

- **MLflow** (versionado, comparación de runs, retraining mensual): `docs/MLFLOW.md`.
- **Ética, sesgos, privacidad y gobernanza**: `docs/ETHICS_GOVERNANCE.md`.
- **Explicabilidad SHAP**: `src/evaluation/shap_analysis.py` + notebook Sprint 4.

## Notebooks

- `notebooks/Sprint1.ipynb` … `Sprint3.ipynb` — EDA, pipeline, modelo final.
- `notebooks/Sprint4_Integracion.ipynb` — SHAP, métricas de negocio/ROI y gobernanza.

## Tests

```bash
pytest
```
