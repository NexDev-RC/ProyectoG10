"""
Paso 7 – Hiperparametrización con Optuna.

Optimiza los hiperparámetros del modelo final (LightGBM por defecto)
usando validación cruzada temporal (TimeSeriesSplit).

Métricas objetivo: RMSE o MAPE (configurable).
"""
import numpy as np
import pandas as pd
import optuna
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import rmse, mape


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────
#  Objetivo de optimización – LightGBM
# ─────────────────────────────────────────────────────────────
def _lgbm_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    metric: str = "rmse",
    n_splits: int = 3,
) -> float:
    """Función objetivo para Optuna con LightGBM."""
    import lightgbm as lgb

    lgb_cfg = cfg["lightgbm"]

    params = {
        "objective":          "regression",
        "metric":             "rmse",
        "verbosity":          -1,
        "n_estimators":       trial.suggest_int("n_estimators",   *lgb_cfg["n_estimators"]),
        "learning_rate":      trial.suggest_float("learning_rate", *lgb_cfg["learning_rate"], log=True),
        "num_leaves":         trial.suggest_int("num_leaves",      *lgb_cfg["num_leaves"]),
        "max_depth":          trial.suggest_int("max_depth",       *lgb_cfg["max_depth"]),
        "min_child_samples":  trial.suggest_int("min_child_samples", *lgb_cfg["min_child_samples"]),
        "subsample":          trial.suggest_float("subsample",     *lgb_cfg["subsample"]),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", *lgb_cfg["colsample_bytree"]),
        "reg_alpha":          trial.suggest_float("reg_alpha",     *lgb_cfg["reg_alpha"]),
        "reg_lambda":         trial.suggest_float("reg_lambda",    *lgb_cfg["reg_lambda"]),
        "random_state":       42,
    }

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        y_pred = model.predict(X_v)

        score = rmse(y_v, y_pred) if metric == "rmse" else mape(y_v, y_pred)
        scores.append(score)

    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
#  Hiperparametrización de LightGBM
# ─────────────────────────────────────────────────────────────
def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list,
    cfg: dict
) -> dict:
    """
    Optimiza hiperparámetros de LightGBM con Optuna.

    Retorna dict con los mejores hiperparámetros.
    """
    optuna_cfg = cfg["optuna"]
    metric     = optuna_cfg.get("metric", "rmse")

    X = X_train[feature_cols].fillna(0).values
    y = y_train.values

    logger.info(
        f"Iniciando Optuna (n_trials={optuna_cfg['n_trials']}, "
        f"timeout={optuna_cfg['timeout']}s, metric={metric})…"
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: _lgbm_objective(
            trial, X, y, cfg,
            metric=metric,
            n_splits=optuna_cfg.get("cv_splits", 3)
        ),
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg["timeout"],
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value  = study.best_value

    logger.info(f"Mejor {metric.upper()}: {best_value:.4f}")
    logger.info(f"Mejores parámetros: {best_params}")

    return {
        "best_params": best_params,
        "best_value":  best_value,
        "study":       study,
    }


# ─────────────────────────────────────────────────────────────
#  Entrenamiento final con mejores hiperparámetros
# ─────────────────────────────────────────────────────────────
def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    feature_cols: list,
    horizon: int = 3,
    cfg: dict | None = None,
) -> "LightGBMForecaster":
    """
    Entrena el modelo final LightGBM con los mejores hiperparámetros.

    Retorna forecaster listo para predicción.
    """
    from src.models.forecaster import LightGBMForecaster

    params = {
        "objective":       "regression",
        "metric":          "rmse",
        "verbosity":       -1,
        "random_state":    42,
        **best_params
    }

    forecaster = LightGBMForecaster(params=params, horizon=horizon)
    forecaster.fit(X_train, y_train, feature_cols=feature_cols)

    logger.info("Modelo final entrenado con mejores hiperparámetros")
    return forecaster


# ─────────────────────────────────────────────────────────────
#  Visualización de resultados Optuna
# ─────────────────────────────────────────────────────────────
def plot_optimization_history(study) -> None:
    """Muestra el historial de optimización Optuna (requiere plotly)."""
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.show()
        fig2 = vis.plot_param_importances(study)
        fig2.show()
    except Exception as e:
        logger.warning(f"No se pudo graficar optimización: {e}")
