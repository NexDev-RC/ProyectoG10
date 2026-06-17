"""
Schemas Pydantic para la API FastAPI.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ForecastPoint(BaseModel):
    date:       str   = Field(..., example="2018-10")
    yhat:       float = Field(..., description="Predicción central (BRL)")
    yhat_lower: float = Field(..., description="Límite inferior (BRL)")
    yhat_upper: float = Field(..., description="Límite superior (BRL)")


class ForecastResponse(BaseModel):
    model:    str
    horizon:  int
    forecast: list[ForecastPoint]
    currency: str = "BRL"


class MetricsResponse(BaseModel):
    model:        str
    rmse:         float
    mape:         float
    mae:          float
    smape:        float
    meets_target: bool
    mape_target:  float = 10.0


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dict con valores de features. Las no especificadas se imputan a 0.",
        example={
            "monthly_orders": 5000,
            "avg_ticket": 150.0,
            "revenue_lag_1": 1200000.0,
            "revenue_rolling_mean_3": 1150000.0,
        }
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    horizon:     int
    currency:    str = "BRL"


class FeatureImportanceResponse(BaseModel):
    features:    list[str]
    importances: list[float]


class ShapResponse(BaseModel):
    """Importancia global SHAP (media del valor absoluto por feature)."""
    horizon:       int   = Field(..., description="Horizonte explicado (meses)")
    base_value:    float = Field(..., description="Valor esperado del modelo (BRL)")
    features:      list[str]
    mean_abs_shap: list[float] = Field(..., description="Media |SHAP| por feature")
