"""
Paso 5 – Limpieza de la Master Table mensual.

Según el flujo del Excel:
  1. Clipado        : recorte de outliers por cuantiles
  2. NaN             : imputación de valores faltantes
  3. Agrupamiento   : categorías raras → "other"
  4. Encoding       : codificación one-hot de variables categóricas restantes
  5. Escalado       : normalización/estandarización de variables numéricas

El resultado es la "MT Limpiada" lista para selección de variables.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
#  Encoding de variables categóricas
# ─────────────────────────────────────────────────────────────
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encoding determinístico para columnas categóricas (object/category).

    A diferencia de `pd.get_dummies`, memoriza en fit() las categorías
    observadas para garantizar columnas consistentes entre train y predict
    (categorías nuevas en transform() se ignoran; categorías ausentes
    quedan en 0).

    Si la tabla no tiene columnas categóricas (caso de la Master Table
    mensual, ya 100% numérica), actúa como passthrough sin modificar nada.
    """

    def __init__(self, cols=None, drop_first: bool = False):
        self.cols = cols
        self.drop_first = drop_first
        self.categories_: dict = {}
        self.cat_cols_: list = []

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_cols_ = self.cols or X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categories_ = {}
        for col in self.cat_cols_:
            cats = sorted(X[col].astype(str).unique().tolist())
            if self.drop_first and len(cats) > 1:
                cats = cats[1:]
            self.categories_[col] = cats
        logger.info(f"CategoricalEncoder ajustado en {len(self.cat_cols_)} columnas categóricas")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, cats in self.categories_.items():
            if col not in X.columns:
                continue
            col_str = X[col].astype(str)
            for cat in cats:
                X[f"{col}__{cat}"] = (col_str == cat).astype(int)
            X = X.drop(columns=[col])
        return X


# ─────────────────────────────────────────────────────────────
#  Escalado / normalización de variables numéricas
# ─────────────────────────────────────────────────────────────
class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Escala columnas numéricas continuas con StandardScaler o MinMaxScaler.

    Parámetros
    ----------
    method         : "standard" | "minmax" | "none"
    cols           : columnas a escalar (None = numéricas con > 2 valores únicos)
    exclude_binary : excluye columnas binarias (flags 0/1) del escalado,
                     ya que no aportan al normalizar/estandarizar.

    Nota: LightGBM (modelo final) es invariante a escala, pero este paso
    deja la "MT Limpiada" lista para modelos sensibles a escala
    (regresión lineal, Prophet con regresores, redes neuronales, etc.)
    y cumple con la actividad de "normalización/escalado" de Sprint 2.
    """

    def __init__(self, method: str = "standard", cols=None, exclude_binary: bool = True):
        self.method = method
        self.cols = cols
        self.exclude_binary = exclude_binary
        self.scaler_ = None
        self.scale_cols_: list = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.method == "none":
            self.scale_cols_ = []
            return self

        cols = self.cols or X.select_dtypes(include="number").columns.tolist()
        if self.exclude_binary:
            cols = [c for c in cols if X[c].dropna().nunique() > 2]
        self.scale_cols_ = cols

        if self.method == "standard":
            self.scaler_ = StandardScaler()
        elif self.method == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Método de escalado desconocido: {self.method}")

        if self.scale_cols_:
            self.scaler_.fit(X[self.scale_cols_])

        logger.info(f"FeatureScaler ({self.method}) ajustado en {len(self.scale_cols_)} columnas")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        if self.method == "none" or not self.scale_cols_:
            return X
        X[self.scale_cols_] = self.scaler_.transform(X[self.scale_cols_])
        return X


# ─────────────────────────────────────────────────────────────
#  Pipeline de limpieza completo
# ─────────────────────────────────────────────────────────────
def build_cleaning_pipeline(cfg: dict):
    """
    Construye el pipeline sklearn de limpieza.

    Pasos (en orden):
      1. clip    : recorte de outliers
      2. impute  : imputación de NaN
      3. grouper : agrupamiento de categorías raras
      4. encode  : one-hot encoding de categóricas restantes
      5. scale   : normalización/estandarización de numéricas

    Retorna sklearn Pipeline reproducible y serializable (joblib).
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
        ("encode",  CategoricalEncoder()),
        ("scale",   FeatureScaler(method=clean_cfg.get("scaling_method", "standard"))),
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
        X_clean = pipeline.fit_transform(X)
        logger.info("Limpieza aplicada (fit+transform)")
    else:
        if pipeline is None:
            raise ValueError("Se requiere un pipeline ajustado cuando fit=False")
        X_clean = pipeline.transform(X)
        logger.info("Limpieza aplicada (transform)")

    # Cada transformer del pipeline retorna un DataFrame, pero las columnas
    # pueden variar respecto a feature_cols si CategoricalEncoder expandió
    # variables categóricas (one-hot). Por eso no forzamos `columns=feature_cols`.
    if not isinstance(X_clean, pd.DataFrame):
        X_clean = pd.DataFrame(X_clean, columns=feature_cols, index=monthly.index)
    else:
        X_clean.index = monthly.index

    result = monthly[list(exclude)].copy()
    result = pd.concat([result, X_clean], axis=1)
    return result, pipeline
