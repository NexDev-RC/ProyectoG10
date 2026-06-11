"""
Paso 8 – Modelos de Forecast.

Implementa tres forecasters candidatos:
  1. LightGBM   : modelo de gradient boosting con features de lags
  2. Prophet    : modelo aditivo de Facebook para series temporales
  3. SARIMA/X   : modelo estadístico clásico con statsmodels

Cada modelo expone la interfaz .fit() / .predict() / .predict_with_intervals().
"""
import warnings
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  LightGBM Forecaster
# ─────────────────────────────────────────────────────────────
class LightGBMForecaster:
    """
    Forecaster basado en LightGBM usando features de lag como entradas.

    Estrategia: Direct multi-step (un modelo por horizonte).
    Para forecasting univariado o multivariado.
    """

    name = "LightGBM"

    def __init__(self, params: dict | None = None, horizon: int = 3):
        import lightgbm as lgb
        self.horizon = horizon
        self.params  = params or {
            "objective":       "regression",
            "metric":          "rmse",
            "n_estimators":    300,
            "learning_rate":   0.05,
            "num_leaves":      31,
            "max_depth":       5,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "random_state":    42,
            "verbose":         -1,
        }
        self.models_: dict = {}
        self.feature_cols_: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        feature_cols: list | None = None
    ):
        import lightgbm as lgb

        self.feature_cols_ = feature_cols or X_train.columns.tolist()
        X = X_train[self.feature_cols_].fillna(0).values

        for h in range(1, self.horizon + 1):
            y_shifted = y_train.shift(-h + 1) if h > 1 else y_train
            # Para pasos futuros, alineamos con los datos disponibles
            mask = ~y_shifted.isna()
            model = lgb.LGBMRegressor(**self.params)

            if X_val is not None and y_val is not None:
                X_v = X_val[self.feature_cols_].fillna(0).values
                model.fit(
                    X[mask], y_shifted[mask],
                    eval_set=[(X_v, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)]
                )
            else:
                model.fit(X[mask], y_shifted[mask])

            self.models_[h] = model

        logger.info(f"LightGBM entrenado ({self.horizon} horizontes)")
        return self

    def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
        """Predice los próximos self.horizon meses."""
        X = X_pred[self.feature_cols_].fillna(0).values
        preds = []
        for h in range(1, self.horizon + 1):
            pred = self.models_[h].predict(X[-1:])
            preds.append(float(pred[0]))
        return np.array(preds)

    def update(self, X_new: pd.DataFrame, y_new: pd.Series, n_extra_trees: int = 20):
        """
        Actualización incremental (warm start) del modelo con datos de un
        nuevo mes (Sprint 2 – simulación de llegada mensual de datos).

        En lugar de reentrenar desde cero, continúa el boosting de cada
        modelo de horizonte a partir de sus árboles existentes vía la API
        nativa `lgb.train(..., init_model=...)`, añadiendo `n_extra_trees`
        árboles ajustados a la(s) nueva(s) observación(es). Esto es mucho
        más rápido que `train_final_model` y refleja un escenario realista
        de actualización mensual.

        Nota: a diferencia del wrapper sklearn (`LGBMRegressor`), la API
        nativa `lgb.train` no exige un mínimo estricto de filas, pero con
        un único dato no hay forma de evaluar una partición y el continue
        training resulta en 0 árboles nuevos (no-op). Se recomienda llamar
        a `update()` con una pequeña ventana de meses recientes (>= 2 filas)
        en lugar de una única observación.

        Parámetros
        ----------
        X_new         : features del/los nuevo(s) mes(es) (ventana reciente,
                          idealmente >= 2 filas)
        y_new         : target real del/los nuevo(s) mes(es)
        n_extra_trees : número de árboles adicionales por horizonte

        Retorna
        -------
        self (modelos actualizados in-place)
        """
        import lightgbm as lgb

        X = X_new[self.feature_cols_].fillna(0).values
        y = y_new.values if hasattr(y_new, "values") else np.asarray(y_new)

        if len(y) == 0:
            logger.warning("update(): no hay observaciones nuevas, se omite la actualización")
            return self

        # Parámetros válidos para la API nativa lgb.train (sin n_estimators,
        # que se controla con num_boost_round).
        native_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        native_params.setdefault("verbosity", -1)

        for h in range(1, self.horizon + 1):
            prev_model = self.models_.get(h)
            init_booster = prev_model.booster_ if prev_model is not None else None

            train_set = lgb.Dataset(X, label=y)
            booster = lgb.train(
                native_params,
                train_set,
                num_boost_round=n_extra_trees,
                init_model=init_booster,
            )
            self.models_[h] = _BoosterModel(booster)

        logger.info(
            f"LightGBM actualizado incrementalmente con {len(y)} observación(es) nueva(s) "
            f"(+{n_extra_trees} árboles por horizonte)"
        )
        return self


    def feature_importance(self) -> pd.DataFrame:
        """Importancia de features del modelo del horizonte 1."""
        model = self.models_.get(1)
        if model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    self.feature_cols_,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)


class _BoosterModel:
    """
    Wrapper ligero sobre `lgb.Booster` para exponer la misma interfaz
    (.predict(), .booster_, .feature_importances_) que `LGBMRegressor`,
    de forma que `LightGBMForecaster` pueda usar indistintamente modelos
    entrenados desde cero o actualizados incrementalmente.
    """

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(np.asarray(X))

    @property
    def feature_importances_(self):
        return np.array(self.booster_.feature_importance())



# ─────────────────────────────────────────────────────────────
#  Prophet Forecaster
# ─────────────────────────────────────────────────────────────
class ProphetForecaster:
    """
    Forecaster basado en Prophet (Meta/Facebook).

    Soporta regressores adicionales (covariables).
    """

    name = "Prophet"

    def __init__(self, params: dict | None = None, horizon: int = 3):
        self.horizon = horizon
        self.params  = params or {
            "seasonality_mode":          "multiplicative",
            "yearly_seasonality":        True,
            "weekly_seasonality":        False,
            "daily_seasonality":         False,
            "changepoint_prior_scale":   0.05,
            "seasonality_prior_scale":   10.0,
        }
        self.model_ = None
        self.regressors_: list = []

    def fit(
        self,
        monthly: pd.DataFrame,
        target_col: str = "monthly_revenue",
        regressors: list | None = None
    ):
        from prophet import Prophet

        df_prophet = monthly[["ds", target_col]].rename(columns={target_col: "y"})

        self.model_ = Prophet(**self.params)

        if regressors:
            self.regressors_ = regressors
            for reg in regressors:
                self.model_.add_regressor(reg)
                df_prophet[reg] = monthly[reg].values

        self.model_.fit(df_prophet)
        logger.info(f"Prophet entrenado con {len(monthly)} puntos")
        return self

    def predict(
        self,
        future_regressors: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Predice los próximos self.horizon meses.

        Retorna DataFrame con columnas: ds, yhat, yhat_lower, yhat_upper.
        """
        future = self.model_.make_future_dataframe(
            periods=self.horizon, freq="MS"
        )

        if self.regressors_ and future_regressors is not None:
            for reg in self.regressors_:
                future[reg] = pd.concat([
                    pd.Series(self.model_.history[reg].values),
                    pd.Series(future_regressors[reg].values)
                ]).values

        forecast = self.model_.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(self.horizon)


# ─────────────────────────────────────────────────────────────
#  SARIMA Forecaster
# ─────────────────────────────────────────────────────────────
class SARIMAForecaster:
    """
    Forecaster SARIMA/SARIMAX usando statsmodels.

    Soporta búsqueda automática de orden con pmdarima (auto_arima).
    """

    name = "SARIMA"

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        auto_arima: bool = True,
        horizon: int = 3
    ):
        self.order         = order
        self.seasonal_order = seasonal_order
        self.auto_arima    = auto_arima
        self.horizon       = horizon
        self.model_        = None
        self.result_       = None

    def fit(self, y_train: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if self.auto_arima:
            try:
                import pmdarima as pm
                auto = pm.auto_arima(
                    y_train,
                    seasonal=True, m=12,
                    information_criterion="aic",
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore"
                )
                self.order         = auto.order
                self.seasonal_order = auto.seasonal_order
                logger.info(f"auto_arima orden: {self.order} x {self.seasonal_order}")
            except ImportError:
                logger.warning("pmdarima no disponible, usando orden por defecto")

        self.model_ = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.result_ = self.model_.fit(disp=False)
        logger.info(f"SARIMA entrenado: AIC={self.result_.aic:.2f}")
        return self

    def predict(self) -> np.ndarray:
        """Predice los próximos self.horizon meses."""
        forecast = self.result_.forecast(steps=self.horizon)
        return np.array(forecast)

    def predict_with_intervals(self, alpha: float = 0.05) -> pd.DataFrame:
        """Predice con intervalos de confianza."""
        pred = self.result_.get_forecast(steps=self.horizon)
        ci   = pred.conf_int(alpha=alpha)
        return pd.DataFrame({
            "yhat":        pred.predicted_mean.values,
            "yhat_lower":  ci.iloc[:, 0].values,
            "yhat_upper":  ci.iloc[:, 1].values,
        })


# ─────────────────────────────────────────────────────────────
#  Ensemble simple (promedio)
# ─────────────────────────────────────────────────────────────
def ensemble_predict(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float] | None = None
) -> np.ndarray:
    """
    Combina predicciones de múltiples modelos.

    Parámetros
    ----------
    predictions : dict {nombre_modelo: array_predicciones}
    weights     : dict {nombre_modelo: peso}  (None = promedio simple)

    Retorna
    -------
    Array con predicciones ensambladas.
    """
    names  = list(predictions.keys())
    arrays = list(predictions.values())
    n_steps = len(arrays[0])

    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    ensemble = np.zeros(n_steps)
    for name, arr in predictions.items():
        ensemble += weights.get(name, 0.0) * np.array(arr)

    logger.info(f"Ensemble: {list(weights.keys())} → pesos {list(weights.values())}")
    return ensemble
