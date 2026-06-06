"""
Paso 5 – Limpieza de la Master Table mensual.

Según el flujo del Excel:
  1. Clipado        : recorte de outliers por cuantiles
  2. NaN             : imputación de valores faltantes
  3. Agrupamiento   : categorías raras → "other"

El resultado es la "MT Limpiada" lista para selección de variables.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


# ─────────────────────────────────────────────────────────────
#  Clipado de outliers
# ─────────────────────────────────────────────────────────────
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Recorta valores extremos de columnas numéricas a los cuantiles
    [q_low, q_high] calculados en fit().

    Parámetros
    ----------
    q_low  : cuantil inferior (ej. 0.01)
    q_high : cuantil superior (ej. 0.99)
    cols   : columnas a clipar (None = todas las numéricas)
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99, cols=None):
        self.q_low  = q_low
        self.q_high = q_high
        self.cols   = cols
        self.bounds_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols or X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            lo = X[col].quantile(self.q_low)
            hi = X[col].quantile(self.q_high)
            self.bounds_[col] = (lo, hi)
        logger.info(f"OutlierClipper ajustado en {len(self.bounds_)} columnas")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                n_clipped = ((X[col] < lo) | (X[col] > hi)).sum()
                X[col] = X[col].clip(lo, hi)
                if n_clipped > 0:
                    logger.debug(f"  {col}: {n_clipped} valores clipados")
        return X


# ─────────────────────────────────────────────────────────────
#  Imputación de NaN
# ─────────────────────────────────────────────────────────────
class NaNImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes.

    Estrategias:
      median       : mediana por columna (recomendado para series temporales)
      mean         : media
      forward_fill : propagación hacia adelante (útil en lags)
      zero         : reemplaza con 0
    """

    def __init__(self, strategy: str = "median", cols=None):
        self.strategy = strategy
        self.cols     = cols
        self.fill_values_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols or X.select_dtypes(include="number").columns.tolist()
        if self.strategy == "median":
            for col in cols:
                self.fill_values_[col] = X[col].median()
        elif self.strategy == "mean":
            for col in cols:
                self.fill_values_[col] = X[col].mean()
        # forward_fill y zero no necesitan fit
        logger.info(f"NaNImputer ajustado ({self.strategy}) en {len(cols)} columnas")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        null_before = X.isnull().sum().sum()

        if self.strategy in ("median", "mean"):
            X = X.fillna(self.fill_values_)
        elif self.strategy == "forward_fill":
            X = X.ffill().bfill()
        elif self.strategy == "zero":
            X = X.fillna(0)

        null_after = X.isnull().sum().sum()
        logger.info(f"NaN imputados: {null_before} → {null_after}")
        return X


# ─────────────────────────────────────────────────────────────
#  Agrupamiento de categorías raras
# ─────────────────────────────────────────────────────────────
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías con frecuencia < threshold en "other".

    Aplica solo a columnas categóricas (object/category).
    """

    def __init__(self, threshold: float = 0.02, cols=None):
        self.threshold = threshold
        self.cols      = cols
        self.keep_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols or X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cols:
            freq = X[col].value_counts(normalize=True)
            self.keep_[col] = freq[freq >= self.threshold].index.tolist()
        logger.info(f"RareCategoryGrouper ajustado en {len(self.keep_)} columnas")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, keep_vals in self.keep_.items():
            if col in X.columns:
                mask = ~X[col].isin(keep_vals)
                X.loc[mask, col] = "other"
        return X


# ─────────────────────────────────────────────────────────────
#  Pipeline de limpieza completo
# ─────────────────────────────────────────────────────────────
def build_cleaning_pipeline(cfg: dict):
    """
    Construye el pipeline sklearn de limpieza.

    Retorna sklearn Pipeline con los tres pasos.
    """
    from sklearn.pipeline import Pipeline

    clean_cfg = cfg["cleaning"]

    pipeline = Pipeline([
        ("clip",    OutlierClipper(
            q_low=clean_cfg["clip_quantile_low"],
            q_high=clean_cfg["clip_quantile_high"]
        )),
        ("impute",  NaNImputer(strategy=clean_cfg["nan_fill_strategy"])),
        ("grouper", RareCategoryGrouper(threshold=clean_cfg["group_rare_threshold"])),
    ])
    return pipeline


def clean_monthly_table(
    monthly: pd.DataFrame,
    cfg: dict,
    fit: bool = True,
    pipeline=None
) -> tuple[pd.DataFrame, object]:
    """
    Aplica el pipeline de limpieza a la tabla mensual.

    Parámetros
    ----------
    monthly  : tabla mensual con features
    cfg      : configuración
    fit      : True = fit+transform (train), False = solo transform (predict)
    pipeline : pipeline pre-ajustado (requerido si fit=False)

    Retorna
    -------
    (tabla_limpia, pipeline_ajustado)
    """
    # Columnas a NO limpiar
    exclude = {"year_month", "ds", "monthly_revenue"}
    feature_cols = [c for c in monthly.columns if c not in exclude]

    X = monthly[feature_cols]

    if fit:
        pipeline = build_cleaning_pipeline(cfg)
        X_clean = pd.DataFrame(
            pipeline.fit_transform(X),
            columns=feature_cols,
            index=monthly.index
        )
        logger.info("Limpieza aplicada (fit+transform)")
    else:
        if pipeline is None:
            raise ValueError("Se requiere un pipeline ajustado cuando fit=False")
        X_clean = pd.DataFrame(
            pipeline.transform(X),
            columns=feature_cols,
            index=monthly.index
        )
        logger.info("Limpieza aplicada (transform)")

    result = monthly[list(exclude)].copy()
    result = pd.concat([result, X_clean], axis=1)
    return result, pipeline
