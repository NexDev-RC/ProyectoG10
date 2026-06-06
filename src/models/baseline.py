"""
Modelos Baseline – referencia antes de modelos avanzados.

Implementa:
  1. Naive            : último valor observado
  2. Moving Average 3M: promedio de los últimos 3 meses
  3. Seasonal Naive   : mismo mes del año anterior
  4. Linear Trend     : regresión lineal sobre el tiempo
"""
import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import compute_technical_metrics


class NaiveModel:
    """Predice el último valor observado para todos los pasos futuros."""

    name = "Naive"

    def fit(self, y_train: np.ndarray):
        self.last_value_ = float(y_train[-1])
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.full(n_steps, self.last_value_)


class MovingAverageModel:
    """Promedio de los últimos `window` meses."""

    def __init__(self, window: int = 3):
        self.window = window
        self.name   = f"MovingAverage_{window}M"

    def fit(self, y_train: np.ndarray):
        self.mean_ = float(np.mean(y_train[-self.window:]))
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.full(n_steps, self.mean_)


class SeasonalNaiveModel:
    """Usa el valor del mismo mes del año anterior (periodo = 12)."""

    name = "SeasonalNaive"

    def fit(self, y_train: np.ndarray):
        self.history_ = np.array(y_train, dtype=float)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        preds = []
        n = len(self.history_)
        for i in range(n_steps):
            idx = n - 12 + i
            val = self.history_[idx] if idx >= 0 else float(np.mean(self.history_))
            preds.append(val)
        return np.array(preds)


class LinearTrendModel:
    """Ajusta tendencia lineal y extrapola."""

    name = "LinearTrend"

    def fit(self, y_train: np.ndarray):
        x = np.arange(len(y_train))
        self.coefs_ = np.polyfit(x, y_train, 1)
        self.n_train_ = len(y_train)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        x_pred = np.arange(self.n_train_, self.n_train_ + n_steps)
        return np.polyval(self.coefs_, x_pred)


# ─────────────────────────────────────────────────────────────
#  Evaluación de todos los baselines
# ─────────────────────────────────────────────────────────────
def evaluate_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "monthly_revenue",
    n_test: int = 3
) -> pd.DataFrame:
    """
    Entrena y evalúa todos los modelos baseline.

    Retorna DataFrame con métricas por modelo.
    """
    y_train = train[target_col].values
    y_test  = test[target_col].values[:n_test]

    models = [
        NaiveModel(),
        MovingAverageModel(window=3),
        SeasonalNaiveModel(),
        LinearTrendModel(),
    ]

    results = []
    for model in models:
        model.fit(y_train)
        y_pred = model.predict(n_test)
        metrics = compute_technical_metrics(y_test, y_pred, model_name=model.name)
        results.append(metrics)

    df_results = pd.DataFrame(results).sort_values("mape")
    best = df_results.iloc[0]

    logger.info(f"\nMejor baseline: {best['model']} (MAPE={best['mape']:.2f}%)")
    logger.info(f"Objetivo final: MAPE < 10%")

    return df_results
