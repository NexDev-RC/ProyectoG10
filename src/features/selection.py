"""
Paso 6 – Selección de Variables.

Implementa el flujo del Excel en orden estricto:

  Orden 0 : Estado Inicial            (todas las features)
  Orden 1 : missing_variable_method   (eliminar features con alta tasa de NaN)
  Orden 2 : PSI_method                (Population Stability Index)
  Orden 3-6: correlation_method       (correlación con umbral paramétrico)
  Orden 7-9: univariante_methods      (información mutua / correlación con target)
  Final    : WRAPPER (RFECV)          (selección recursiva con validación cruzada)

El resultado es la lista final de features seleccionadas + tabla de trazabilidad.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    RFECV,
    mutual_info_regression,
    SelectKBest,
    f_regression,
)
from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────
#  PSI – Population Stability Index
# ─────────────────────────────────────────────────────────────
def calculate_psi(
    base: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6
) -> float:
    """
    Calcula PSI entre una distribución base (train) y actual (val/test).

    PSI < 0.10  → estable
    PSI 0.10-0.20 → monitorear
    PSI > 0.20  → inestable (eliminar feature)
    """
    base    = base.dropna()
    current = current.dropna()

    if len(base) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.linspace(base.min(), base.max(), bins + 1)
    breakpoints[0]  -= 1e-9
    breakpoints[-1] += 1e-9

    base_pct = np.histogram(base, bins=breakpoints)[0] / len(base) + epsilon
    curr_pct = np.histogram(current, bins=breakpoints)[0] / len(current) + epsilon

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


def psi_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    threshold: float = 0.20,
    bins: int = 10
) -> tuple[list[str], pd.DataFrame]:
    """Elimina features con PSI > threshold entre train y val."""
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    psi_scores = {}
    for col in num_cols:
        psi_scores[col] = calculate_psi(X_train[col], X_val[col], bins=bins)

    psi_df = pd.DataFrame.from_dict(psi_scores, orient="index", columns=["psi"])
    psi_df["stable"] = psi_df["psi"] <= threshold
    psi_df = psi_df.sort_values("psi", ascending=False)

    selected = psi_df[psi_df["stable"]].index.tolist()
    removed  = psi_df[~psi_df["stable"]].index.tolist()
    logger.info(f"PSI: {len(selected)} features estables, {len(removed)} eliminadas")
    return selected, psi_df


# ─────────────────────────────────────────────────────────────
#  Missing rate selection
# ─────────────────────────────────────────────────────────────
def missing_selection(
    X: pd.DataFrame,
    threshold: float = 0.15
) -> tuple[list[str], pd.DataFrame]:
    """Elimina features con tasa de NaN > threshold."""
    miss_rate = X.isnull().mean()
    miss_df = miss_rate.to_frame("missing_rate")
    miss_df["keep"] = miss_df["missing_rate"] <= threshold

    selected = miss_df[miss_df["keep"]].index.tolist()
    removed  = miss_df[~miss_df["keep"]].index.tolist()
    logger.info(f"Missing: {len(selected)} features OK, {len(removed)} eliminadas")
    return selected, miss_df


# ─────────────────────────────────────────────────────────────
#  Correlación (entre features)
# ─────────────────────────────────────────────────────────────
def correlation_selection(
    X: pd.DataFrame,
    threshold: float = 0.95
) -> tuple[list[str], pd.DataFrame]:
    """Elimina features con correlación > threshold (mantiene la primera de cada par)."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    corr_matrix = X[num_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    selected = [c for c in num_cols if c not in to_drop]
    logger.info(
        f"Correlación (umbral={threshold}): "
        f"{len(selected)} features OK, {len(to_drop)} eliminadas"
    )
    return selected, corr_matrix


# ─────────────────────────────────────────────────────────────
#  Univariante – información mutua con el target
# ─────────────────────────────────────────────────────────────
def univariate_selection(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.30,
    min_features: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """
    Elimina features con información mutua normalizada < threshold respecto al target.
    Garantiza retener al menos min_features (las de mayor MI score).
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()

    if len(num_cols) == 0:
        logger.warning("Univariante: no hay columnas numéricas disponibles")
        return [], pd.DataFrame()

    X_num = X[num_cols].fillna(0)

    if len(X_num) < 2:
        logger.warning("Univariante: pocas muestras, se retienen todas las features")
        return num_cols, pd.DataFrame(
            {"mi_score": [1.0]*len(num_cols),
             "mi_normalized": [1.0]*len(num_cols),
             "keep": [True]*len(num_cols)},
            index=num_cols
        )

    mi_scores = mutual_info_regression(X_num, y, random_state=42)
    max_mi = mi_scores.max() if mi_scores.max() > 0 else 1.0
    mi_normalized = mi_scores / max_mi

    scores_df = pd.DataFrame({
        "feature": num_cols,
        "mi_score": mi_scores,
        "mi_normalized": mi_normalized,
    }).set_index("feature").sort_values("mi_normalized", ascending=False)

    scores_df["keep"] = scores_df["mi_normalized"] >= threshold

    selected = scores_df[scores_df["keep"]].index.tolist()
    if len(selected) < min_features:
        top_n = min(min_features, len(num_cols))
        selected = scores_df.head(top_n).index.tolist()
        scores_df["keep"] = scores_df.index.isin(selected)
        logger.warning(
            f"Univariante: umbral={threshold} dejó pocas features, "
            f"forzando top-{top_n} por MI score"
        )

    removed = [f for f in num_cols if f not in selected]
    logger.info(
        f"Univariante (umbral={threshold}): "
        f"{len(selected)} features OK, {len(removed)} eliminadas"
    )
    return selected, scores_df


# ─────────────────────────────────────────────────────────────
#  WRAPPER – RFECV
# ─────────────────────────────────────────────────────────────
def wrapper_selection(
    X: pd.DataFrame,
    y: pd.Series,
    estimator=None,
    cv: int = 3,
    scoring: str = "neg_root_mean_squared_error",
) -> tuple[list[str], object]:
    """Selección RFECV (Recursive Feature Elimination with CV)."""
    from sklearn.model_selection import TimeSeriesSplit

    if estimator is None:
        estimator = Ridge(alpha=1.0)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    X_num = X[num_cols].fillna(0)

    tscv = TimeSeriesSplit(n_splits=cv)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=tscv,
        scoring=scoring,
        min_features_to_select=3,
        n_jobs=-1,
    )
    rfecv.fit(X_num, y)

    selected = [col for col, flag in zip(num_cols, rfecv.support_) if flag]
    logger.info(f"WRAPPER RFECV: {len(selected)} features seleccionadas")
    return selected, rfecv


# ─────────────────────────────────────────────────────────────
#  Pipeline completo de selección (según Excel)
# ─────────────────────────────────────────────────────────────
MIN_FEATURES = 5  # mínimo absoluto de features a retener en cada paso


def _safe_filter(
    current: list[str],
    proposed: list[str],
    step_name: str,
) -> list[str]:
    """
    Aplica la lista propuesta solo si deja al menos MIN_FEATURES features.
    Si no, conserva las top-MIN_FEATURES del conjunto actual.
    """
    if len(proposed) >= MIN_FEATURES:
        return proposed
    # fallback: retener features del conjunto actual que estén en proposed,
    # completando con las primeras de current hasta MIN_FEATURES
    kept = [f for f in proposed if f in current]
    extras = [f for f in current if f not in kept]
    result = (kept + extras)[:MIN_FEATURES]
    logger.warning(
        f"{step_name}: propuesta dejó {len(proposed)} features "
        f"(< {MIN_FEATURES}), se retienen {len(result)}"
    )
    return result


def run_feature_selection(
    monthly_train: pd.DataFrame,
    monthly_val: pd.DataFrame,
    target_col: str,
    cfg: dict,
    explore: bool = False
) -> dict:
    """
    Ejecuta el pipeline completo de selección de variables.

    Pasos: Missing → PSI → Correlación → Univariante → Wrapper (RFECV)

    Retorna dict con:
      selected_features : lista de features seleccionadas
      steps_report      : dict de resultados por paso
      rfecv             : objeto RFECV ajustado (o None si omitido)
    """
    sel_cfg = cfg["feature_selection"]
    exclude = {target_col, "year_month", "ds", "monthly_revenue"}

    all_features = [c for c in monthly_train.columns if c not in exclude]
    logger.info(f"\n{'='*60}")
    logger.info(f"SELECCIÓN DE VARIABLES – Estado inicial: {len(all_features)} features")
    logger.info(f"{'='*60}")

    X_train = monthly_train[all_features]
    X_val   = monthly_val[[c for c in all_features if c in monthly_val.columns]]
    y_train = monthly_train[target_col]

    current_features = all_features[:]
    steps_report: dict = {
        "0_initial": {"features_remaining": len(current_features)}
    }

    # ── Paso 1: Missing
    sel_miss, miss_df = missing_selection(
        X_train,
        threshold=sel_cfg["missing_threshold"]
    )
    current_features = _safe_filter(current_features, sel_miss, "Missing")
    steps_report["1_missing"] = {
        "threshold": sel_cfg["missing_threshold"],
        "features_remaining": len(current_features),
        "report": miss_df
    }

    # ── Paso 2: PSI (omitir si val es muy pequeño o current_features vacío)
    if len(X_val) >= 3 and len(current_features) > 0:
        sel_psi, psi_df = psi_selection(
            X_train[current_features],
            X_val[[c for c in current_features if c in X_val.columns]],
            threshold=sel_cfg["psi_threshold"],
            bins=min(sel_cfg["psi_bins"], max(2, len(X_val) - 1))
        )
        current_features = _safe_filter(current_features, sel_psi, "PSI")
        steps_report["2_psi"] = {
            "features_remaining": len(current_features),
            "report": psi_df
        }
    else:
        logger.warning(
            f"PSI omitido: val tiene {len(X_val)} muestras (mínimo 3) "
            f"o no hay features disponibles"
        )
        steps_report["2_psi"] = {
            "features_remaining": len(current_features),
            "skipped": True
        }

    # ── Paso 3: Correlación
    if len(current_features) > 0:
        corr_threshold = sel_cfg["correlation_final"]
        sel_corr, corr_matrix = correlation_selection(
            X_train[current_features],
            threshold=corr_threshold
        )
        current_features = _safe_filter(current_features, sel_corr, "Correlación")
        steps_report["3_correlation"] = {
            "threshold": corr_threshold,
            "features_remaining": len(current_features),
            "report": corr_matrix
        }

    # ── Paso 4: Univariante
    if len(current_features) > 0:
        uni_threshold = sel_cfg["univariate_final"]
        sel_uni, uni_df = univariate_selection(
            X_train[current_features],
            y_train,
            threshold=uni_threshold,
            min_features=MIN_FEATURES
        )
        current_features = _safe_filter(current_features, sel_uni, "Univariante")
        steps_report["4_univariate"] = {
            "threshold": uni_threshold,
            "features_remaining": len(current_features),
            "report": uni_df
        }

    # ── Paso 5: Wrapper RFECV
    n_splits_wrapper = sel_cfg.get("wrapper_cv", 3)
    min_samples_wrapper = n_splits_wrapper + 1
    if len(current_features) >= 3 and len(X_train) >= min_samples_wrapper:
        sel_wrap, rfecv = wrapper_selection(
            X_train[current_features],
            y_train,
            cv=n_splits_wrapper
        )
        current_features = _safe_filter(current_features, sel_wrap, "Wrapper")
        steps_report["5_wrapper"] = {
            "features_remaining": len(current_features),
            "rfecv": rfecv
        }
    else:
        rfecv = None
        logger.warning(
            f"Wrapper omitido: {len(current_features)} features o "
            f"{len(X_train)} muestras insuficientes (mínimo {min_samples_wrapper})"
        )
        steps_report["5_wrapper"] = {
            "features_remaining": len(current_features),
            "skipped": True
        }

    # ── Resumen
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN DE SELECCIÓN DE VARIABLES:")
    for step, info in steps_report.items():
        logger.info(f"  {step}: {info['features_remaining']} features")
    logger.info(f"  → Features finales seleccionadas: {len(current_features)}")
    logger.info(f"{'='*60}")

    return {
        "selected_features": current_features,
        "steps_report":      steps_report,
        "rfecv":             rfecv,
    }


def get_selection_summary(steps_report: dict) -> pd.DataFrame:
    """Genera tabla resumen del proceso de selección."""
    rows = []
    for step, info in steps_report.items():
        row = {
            "step": step,
            "features_remaining": info.get("features_remaining", "N/A"),
        }
        if "threshold" in info:
            row["threshold"] = info["threshold"]
        if info.get("skipped"):
            row["note"] = "omitido"
        rows.append(row)
    return pd.DataFrame(rows)