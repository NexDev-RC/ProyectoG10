"""
Tabla de trazabilidad de la Selección de Variables (Sprint 3).

Replica la "Salida de la Selección de Variables" del flujo de ejemplo:

  | Nro | Dominio | Variable | flagSelected | importancia_seleccion | importancia_modelo_final |

Adaptación a series temporales:
  - Dominio                  : inferido del nombre de la variable (en el ejemplo
                                eran tablas Olist: Order Items, Población Objetivo…)
  - flagSelected             : 1 si la variable sobrevivió al flujo
                                Missing → PSI → Correlación → Univariante → Wrapper
  - importancia_seleccion    : score de información mutua con el target normalizado
                                a 100% entre las seleccionadas (equivalente al
                                11.00% / 8.76% / 1.05% del ejemplo)
  - importancia_modelo_final : importancia en el modelo final entrenado,
                                normalizada a 100%
"""
import numpy as np
import pandas as pd
from loguru import logger


# ─── Reglas de dominio. La coincidencia EXACTA tiene prioridad sobre el
#     prefijo (evita que p. ej. 'monthly_orders' caiga en Calendario por
#     empezar con 'month').
_DOMAIN_RULES = [
    (("monthly_revenue",),                                    "Target"),
    (("monthly_orders", "monthly_items"),                     "Órdenes"),
    (("monthly_customers",),                                  "Clientes"),
    (("revenue_lag", "revenue_rolling", "revenue_mom",
      "revenue_yoy"),                                         "Historia del Target (lags)"),
    (("orders_lag",),                                         "Historia de Órdenes (lags)"),
    (("ds", "year_month", "month", "quarter", "year",
      "month_sin", "month_cos", "is_q4", "is_nov_dec"),       "Calendario / Estacionalidad"),
    (("avg_ticket", "median_ticket", "avg_price"),            "Ticket / Precio"),
    (("avg_freight", "total_freight"),                        "Flete"),
    (("avg_review_score",),                                   "Satisfacción"),
    (("avg_delivery", "avg_delay", "pct_late"),               "Entregas"),
    (("pct_credit", "pct_boleto", "pct_installments",
      "avg_installments"),                                    "Pagos"),
    (("unique_categories", "unique_sellers",
      "unique_states"),                                       "Diversidad de Oferta"),
]


def infer_domain(col: str) -> str:
    """
    Asigna un dominio de negocio a una variable según su nombre.

    Dos pasadas: primero coincidencia exacta (gana siempre), luego prefijo.
    """
    for prefixes, domain in _DOMAIN_RULES:
        if col in prefixes:
            return domain
    for prefixes, domain in _DOMAIN_RULES:
        for p in prefixes:
            if col.startswith(p):
                return domain
    return "Otros"


def build_selection_table(
    all_columns: list[str],
    selected_features: list[str],
    steps_report: dict,
    final_importance: pd.DataFrame | None = None,
    target_col: str = "monthly_revenue",
) -> pd.DataFrame:
    """
    Construye la tabla de trazabilidad variable a variable.

    Parámetros
    ----------
    all_columns       : todas las columnas de la tabla mensual (incluye target/ds)
    selected_features : features que sobrevivieron a la selección
    steps_report      : dict retornado por run_feature_selection (usa el reporte
                        univariante para la importancia de selección)
    final_importance  : DataFrame [feature, importance] del modelo final entrenado
    target_col        : nombre del target (se marca como dominio Target, flag 0)

    Retorna
    -------
    DataFrame con columnas:
      Nro, Dominio, Variable, flagSelected,
      importancia_seleccion (%), importancia_modelo_final (%)
    """
    # ── Importancia de selección: MI score del paso univariante
    mi_scores = {}
    uni_report = steps_report.get("4_univariate", {}).get("report")
    if uni_report is not None and isinstance(uni_report, pd.DataFrame) and "mi_score" in uni_report:
        mi_scores = uni_report["mi_score"].to_dict()

    sel_total = sum(mi_scores.get(f, 0.0) for f in selected_features)

    # ── Importancia del modelo final
    fi = {}
    if final_importance is not None and len(final_importance) > 0:
        total_imp = final_importance["importance"].sum() or 1.0
        fi = dict(zip(
            final_importance["feature"],
            final_importance["importance"] / total_imp * 100,
        ))

    rows = []
    for i, col in enumerate(all_columns, start=1):
        is_selected = int(col in selected_features)
        if sel_total > 0 and is_selected:
            imp_sel = mi_scores.get(col, 0.0) / sel_total * 100
        else:
            imp_sel = 0.0
        rows.append({
            "Nro":                       i,
            "Dominio":                   infer_domain(col),
            "Variable":                  col,
            "flagSelected":              is_selected,
            "importancia_seleccion":     round(imp_sel, 2),
            "importancia_modelo_final":  round(fi.get(col, 0.0), 2),
        })

    table = pd.DataFrame(rows)

    n_sel = int(table["flagSelected"].sum())
    logger.info(
        f"Tabla de selección generada: {len(table)} variables, "
        f"{n_sel} seleccionadas "
        f"(importancia_seleccion suma {table['importancia_seleccion'].sum():.1f}%)"
    )
    return table


def export_selection_table(table: pd.DataFrame, path: str = "reports/seleccion_variables.csv") -> str:
    """Exporta la tabla a CSV ("Tabla o un csv con variables seleccionadas")."""
    table.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Tabla de selección exportada: {path}")
    return path
