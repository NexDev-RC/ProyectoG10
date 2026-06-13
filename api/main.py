"""
API REST – Olist Revenue Forecast.

Endpoints:
  GET  /api/v1/health            : health check
  GET  /api/v1/forecast          : predicción para los próximos N meses
  GET  /api/v1/metrics           : métricas técnicas y de negocio del modelo
  POST /api/v1/predict           : predicción con datos de entrada custom
  GET  /api/v1/feature-importance: importancia de features del modelo

Iniciar:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import load_config, load_model
from api.schemas import (
    ForecastResponse, ForecastPoint, MetricsResponse,
    PredictRequest, PredictResponse, FeatureImportanceResponse
)


# ─────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────
cfg = load_config()
api_cfg = cfg["api"]

app = FastAPI(
    title=api_cfg["title"],
    version=api_cfg["version"],
    description="API para forecast de ingresos mensuales del e-commerce Olist",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading de modelos
_forecaster    = None
_cleaning_pipe = None
_monthly_cache = None


def get_forecaster():
    global _forecaster
    if _forecaster is None:
        try:
            _forecaster = load_model("lgbm_forecaster", cfg)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Modelo no entrenado. Ejecuta train_pipeline primero."
            )
    return _forecaster


def get_cleaning_pipe():
    global _cleaning_pipe
    if _cleaning_pipe is None:
        try:
            _cleaning_pipe = load_model("cleaning_pipeline", cfg)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Pipeline de limpieza no encontrado."
            )
    return _cleaning_pipe


def get_monthly_data() -> pd.DataFrame:
    global _monthly_cache
    if _monthly_cache is None:
        try:
            from src.utils.helpers import load_dataframe
            _monthly_cache = load_dataframe("monthly_features", cfg)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Datos mensuales no disponibles. Ejecuta train_pipeline primero."
            )
    return _monthly_cache


# ─────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["Sistema"])
def health_check():
    """Verifica que la API esté operativa."""
    return {
        "status":  "ok",
        "version": api_cfg["version"],
        "model":   cfg["models"]["final_model"],
    }


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["Predicción"])
def get_forecast(
    horizon: int = Query(default=3, ge=1, le=12, description="Meses a predecir")
):
    """
    Genera el forecast de revenue para los próximos N meses.

    - **horizon**: número de meses a predecir (1-12, default 3)
    """
    from src.pipeline.predict_pipeline import PredictPipeline

    pipe = PredictPipeline(cfg)
    forecast_df = pipe.run(horizon=horizon)

    points = [
        ForecastPoint(
            date       = row["ds"].strftime("%Y-%m"),
            yhat       = round(row["yhat"], 2),
            yhat_lower = round(row["yhat_lower"], 2),
            yhat_upper = round(row["yhat_upper"], 2),
        )
        for _, row in forecast_df.iterrows()
    ]

    return ForecastResponse(
        model=cfg["models"]["final_model"],
        horizon=horizon,
        forecast=points,
        currency="BRL",
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Métricas"])
def get_metrics():
    """
    Retorna las métricas técnicas (backtest) del modelo campeón actual.

    Lee `metadata['metrics_backtest']` desde `final_model.pkl`, generado
    en el Paso 5 del Sprint 3 (selección y exportación del modelo final).
    """
    mape_target = cfg["metrics"]["target_mape"]
    try:
        bundle = load_model("final_model", cfg)
        metadata = bundle["metadata"]
        metrics = metadata["metrics_backtest"]
        mape = float(metrics["mape"])
        return MetricsResponse(
            model=metadata.get("model_type", cfg["models"]["final_model"]),
            rmse=float(metrics["rmse"]),
            mape=mape,
            mae=float(metrics["mae"]),
            smape=float(metrics["smape"]),
            meets_target=mape < mape_target,
            mape_target=mape_target,
        )
    except (FileNotFoundError, KeyError) as exc:
        logger.warning(f"No se pudo cargar final_model.pkl para /metrics: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Métricas no disponibles. Ejecuta el Paso 5 del Sprint 3 "
                   "(exportación de final_model.pkl) primero.",
        )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Predicción"])
def predict_custom(payload: PredictRequest):
    """
    Genera una predicción con features personalizadas.

    Útil para escenarios hipotéticos ("¿qué pasaría si…?").
    """
    forecaster = get_forecaster()
    features   = forecaster.feature_cols_

    # Construir input
    input_data = {}
    for feat in features:
        input_data[feat] = payload.features.get(feat, 0.0)

    X = pd.DataFrame([input_data])
    y_pred = forecaster.predict(X)

    return PredictResponse(
        predictions=[round(v, 2) for v in y_pred.tolist()],
        horizon=len(y_pred),
        currency="BRL",
    )


@app.get("/api/v1/feature-importance", response_model=FeatureImportanceResponse, tags=["Modelo"])
def get_feature_importance(top_n: int = Query(default=20, ge=5, le=60)):
    """Retorna la importancia de las features del modelo."""
    forecaster = get_forecaster()
    fi_df = forecaster.feature_importance().head(top_n)

    return FeatureImportanceResponse(
        features=fi_df["feature"].tolist(),
        importances=fi_df["importance"].tolist(),
    )


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=True
    )
