"""
Gap de sobreajuste: train vs backtest (Sprint 3).

El flujo de ejemplo indica que la hiperparametrización debe buscar
"reducir la distancia entre AUC train y AUC backtest" — es decir,
controlar el sobreajuste, no solo maximizar la métrica de validación.

Adaptación a regresión / series temporales: se compara el RMSE
in-sample (train) contra el RMSE out-of-sample (backtest) de cada
candidato. Un gap grande = el modelo memoriza el train y generaliza
mal; el modelo elegido debe equilibrar error bajo y gap razonable.

Nota: se usa RMSE (no MAPE) para el lado train porque los primeros
meses de la serie Olist tienen ingresos cercanos a cero y el MAPE
in-sample se vuelve numéricamente absurdo (divisiones por valores
minúsculos). El MAPE se reporta solo en backtest, donde la escala
es estable.
"""
import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import rmse, mape


def _insample_predictions(forecaster, X_train: pd.DataFrame) -> np.ndarray:
    """Predicciones in-sample del modelo de horizonte 1."""
    X = X_train[forecaster.feature_cols_].fillna(0).values
    return np.asarray(forecaster.models_[1].predict(X))


def train_backtest_gap(
    forecaster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    name: str | None = None,
) -> dict:
    """
    Calcula RMSE train (in-sample, h=1) vs backtest y su gap.

    Aplica a forecasters con interfaz models_[h] / feature_cols_
    (LightGBMForecaster, SklearnForecaster).

    Interpretación de gap_rmse_ratio:
      ~1 a 3  : generalización razonable
      >> 3    : sobreajuste fuerte (memoriza el train)
      < 1     : posible con backtest pequeño (n=3) — el error de
                train puede superar al de test por azar; no indica
                un modelo "mejor que perfecto".

    Retorna dict: model, rmse_train, rmse_backtest, gap_rmse_ratio,
                  mape_backtest
    """
    y_fit = _insample_predictions(forecaster, X_train)
    y_tr  = np.asarray(y_train)

    rmse_tr, rmse_bt = rmse(y_tr, y_fit), rmse(y_test, y_pred_test)

    return {
        "model":          name or getattr(forecaster, "name", "model"),
        "rmse_train":     rmse_tr,
        "rmse_backtest":  rmse_bt,
        # cuántas veces crece el error al salir del train (~1 = sin gap)
        "gap_rmse_ratio": rmse_bt / rmse_tr if rmse_tr > 0 else np.inf,
        "mape_backtest":  mape(y_test, y_pred_test),
    }


def gap_summary(gaps: list[dict]) -> pd.DataFrame:
    """Tabla resumen de gaps ordenada por RMSE de backtest (mejor primero)."""
    df = pd.DataFrame(gaps).sort_values("rmse_backtest").reset_index(drop=True)
    logger.info("Resumen de gap train→backtest calculado")
    return df


def selection_cost(
    metrics_all_features: dict,
    metrics_selected: dict,
    n_all: int,
    n_selected: int,
) -> pd.DataFrame:
    """
    Costo (o beneficio) de la selección de variables.

    Equivalente al "60 → 30 features: Gini 50 → 47" del flujo de ejemplo:
    cuantifica cuánta métrica se sacrifica (o gana) al reducir features.

    Parámetros
    ----------
    metrics_all_features : dict con rmse/mape del modelo con TODAS las features
    metrics_selected     : dict con rmse/mape del modelo con las seleccionadas
    n_all, n_selected    : número de features antes/después

    Retorna DataFrame comparativo de 2 filas.
    """
    df = pd.DataFrame([
        {
            "estado":     f"Todas las features ({n_all})",
            "n_features": n_all,
            "rmse":       metrics_all_features["rmse"],
            "mape":       metrics_all_features["mape"],
        },
        {
            "estado":     f"Seleccionadas ({n_selected})",
            "n_features": n_selected,
            "rmse":       metrics_selected["rmse"],
            "mape":       metrics_selected["mape"],
        },
    ])
    delta = df.iloc[1]["mape"] - df.iloc[0]["mape"]
    logger.info(
        f"Costo de selección: {n_all} → {n_selected} features, "
        f"ΔMAPE = {delta:+.2f} pp"
    )
    return df
