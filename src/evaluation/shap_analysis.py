"""
Análisis SHAP – Explicabilidad del modelo de forecast (Sprint 4).

El modelo final es un `LightGBMForecaster`, un *wrapper* que contiene un
estimador por horizonte en `forecaster.models_[h]` (cada uno un
`lgb.LGBMRegressor` o un `_BoosterModel` sobre `lgb.Booster`). SHAP no
explica el wrapper directamente, por lo que se extrae el estimador del
horizonte deseado (por defecto h=1, predicción a 1 mes) y se usa
`shap.TreeExplainer`, nativamente compatible con LightGBM.

Funciones reutilizables por el notebook, la API (`/api/v1/shap`) y el
dashboard.

Uso:
    from src.utils.helpers import load_config, load_model, load_dataframe
    from src.features.cleaning import clean_monthly_table
    from src.evaluation.shap_analysis import compute_shap_values, shap_summary_df

    cfg        = load_config()
    forecaster = load_model("lgbm_forecaster", cfg)
    clean_pipe = load_model("cleaning_pipeline", cfg)
    monthly    = load_dataframe("monthly_features", cfg)
    monthly_clean, _ = clean_monthly_table(monthly, cfg, fit=False, pipeline=clean_pipe)

    X = monthly_clean[forecaster.feature_cols_]
    sv, fnames, base = compute_shap_values(forecaster, X)
    summary = shap_summary_df(sv, fnames)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────
#  Extracción del estimador base (compatible con SHAP)
# ─────────────────────────────────────────────────────────────
def _base_estimator(forecaster, horizon: int = 1):
    """
    Devuelve un objeto que `shap.TreeExplainer` puede explicar.

    `forecaster.models_[h]` puede ser un `LGBMRegressor` (atributo
    `booster_`) o un `_BoosterModel` (atributo `booster_` con el
    `lgb.Booster`). En ambos casos pasamos el `Booster` subyacente,
    que es lo que SHAP maneja de forma robusta.
    """
    if horizon not in getattr(forecaster, "models_", {}):
        raise ValueError(
            f"Horizonte {horizon} no disponible. "
            f"Horizontes entrenados: {list(forecaster.models_.keys())}"
        )
    model = forecaster.models_[horizon]
    booster = getattr(model, "booster_", None)
    return booster if booster is not None else model


def build_explainer(forecaster, horizon: int = 1):
    """Crea un `shap.TreeExplainer` para el modelo del horizonte indicado."""
    import shap

    estimator = _base_estimator(forecaster, horizon)
    explainer = shap.TreeExplainer(estimator)
    logger.info(f"SHAP TreeExplainer creado para horizonte h={horizon}")
    return explainer


def _align_X(forecaster, X: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y ordena las columnas según `forecaster.feature_cols_`."""
    cols = forecaster.feature_cols_
    return X[cols].fillna(0)


# ─────────────────────────────────────────────────────────────
#  Cálculo de valores SHAP
# ─────────────────────────────────────────────────────────────
def compute_shap_values(
    forecaster,
    X: pd.DataFrame,
    horizon: int = 1,
):
    """
    Calcula los valores SHAP para un conjunto de observaciones.

    Parámetros
    ----------
    forecaster : LightGBMForecaster ya entrenado
    X          : DataFrame con (al menos) las columnas de `feature_cols_`
    horizon    : horizonte a explicar (1 = predicción a 1 mes)

    Retorna
    -------
    shap_values   : np.ndarray (n_samples, n_features)
    feature_names : list[str]
    base_value    : float (valor esperado / expected_value del explainer)
    """
    explainer = build_explainer(forecaster, horizon)
    X_aligned = _align_X(forecaster, X)

    shap_values = explainer.shap_values(X_aligned)
    # Algunas versiones devuelven lista (multi-output); tomamos el primero.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[0])
    else:
        base_value = float(base_value)

    feature_names = list(X_aligned.columns)
    logger.info(
        f"SHAP calculado: {shap_values.shape[0]} obs × "
        f"{shap_values.shape[1]} features (base={base_value:,.2f})"
    )
    return shap_values, feature_names, base_value


def shap_summary_df(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Importancia global = media del valor absoluto de SHAP por feature.

    Retorna un DataFrame ordenado descendente con columnas
    `feature`, `mean_abs_shap`.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_contributions_df(
    shap_values: np.ndarray,
    feature_names: list[str],
    index: int = -1,
) -> pd.DataFrame:
    """
    Contribuciones SHAP de una única observación (por defecto la última),
    ordenadas por magnitud. Útil para explicación local en el dashboard.

    Retorna DataFrame con `feature`, `shap_value`.
    """
    row = shap_values[index]
    return (
        pd.DataFrame({"feature": feature_names, "shap_value": row})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────
#  Gráficos guardados a disco (para notebook / reportes)
# ─────────────────────────────────────────────────────────────
def save_shap_plots(
    forecaster,
    X: pd.DataFrame,
    out_dir: str = "reports",
    horizon: int = 1,
    prefix: str = "sprint4_shap",
):
    """
    Genera y guarda los gráficos SHAP estándar:
      - {prefix}_summary.png  : beeswarm (distribución de impactos)
      - {prefix}_bar.png      : importancia global (media |SHAP|)

    Retorna la lista de rutas guardadas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    explainer = build_explainer(forecaster, horizon)
    X_aligned = _align_X(forecaster, X)
    shap_values = explainer.shap_values(X_aligned)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    paths = []

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_aligned, show=False)
    p1 = out / f"{prefix}_summary.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(str(p1))

    # Barra (importancia global)
    plt.figure()
    shap.summary_plot(shap_values, X_aligned, plot_type="bar", show=False)
    p2 = out / f"{prefix}_bar.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(str(p2))

    logger.info(f"Gráficos SHAP guardados: {paths}")
    return paths
