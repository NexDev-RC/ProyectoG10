"""
Forecaster genérico para estimadores sklearn-compatibles (Sprint 3).

Permite comparar los modelos del flujo de ejemplo (Random Forest,
Regresión lineal/Ridge, XGBoost) bajo la MISMA estrategia direct
multi-step que LightGBMForecaster, de modo que la comparación de
modelos sea justa: mismos features, mismos splits, misma forma de
predecir.

Uso:
    from sklearn.ensemble import RandomForestRegressor
    rf = SklearnForecaster(RandomForestRegressor(n_estimators=300), name="RandomForest")
    rf.fit(X_train, y_train, feature_cols=selected)
    y_pred = rf.predict(X_train.tail(1))
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone


class SklearnForecaster:
    """
    Direct multi-step forecasting con cualquier regresor sklearn.

    Entrena un modelo independiente por horizonte h (1..horizon),
    igual que LightGBMForecaster.
    """

    def __init__(self, estimator, name: str = "sklearn", horizon: int = 3):
        self.base_estimator = estimator
        self.name           = name
        self.horizon        = horizon
        self.models_: dict  = {}
        self.feature_cols_: list = []
        self.resid_std_: dict = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list | None = None):
        self.feature_cols_ = feature_cols or X_train.columns.tolist()
        X = X_train[self.feature_cols_].fillna(0).values

        for h in range(1, self.horizon + 1):
            y_shifted = y_train.shift(-h + 1) if h > 1 else y_train
            mask = ~y_shifted.isna()

            model = clone(self.base_estimator)
            model.fit(X[mask], np.asarray(y_shifted[mask]))
            self.models_[h] = model

            resid = np.asarray(y_shifted[mask]) - model.predict(X[mask])
            self.resid_std_[h] = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

        logger.info(f"{self.name} entrenado ({self.horizon} horizontes)")
        return self

    def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
        """Predice los próximos self.horizon meses desde la última fila."""
        X = X_pred[self.feature_cols_].fillna(0).values
        return np.array([
            float(self.models_[h].predict(X[-1:])[0])
            for h in range(1, self.horizon + 1)
        ])

    def feature_importance(self) -> pd.DataFrame:
        """Importancia del modelo de horizonte 1 (si el estimador la expone)."""
        model = self.models_.get(1)
        if model is None:
            return pd.DataFrame()
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        else:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    self.feature_cols_,
            "importance": imp,
        }).sort_values("importance", ascending=False)
