"""
Métricas técnicas y de negocio.

Técnicas : RMSE, MAPE, MAE, sMAPE
Negocio  : revenue_total, avg_monthly, retention_rate,
           delivery_kpis, review_score, cancellation_rate
"""
import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────
#  Métricas técnicas
# ─────────────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Excluye ceros en y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def compute_technical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model"
) -> dict:
    """Calcula todas las métricas técnicas de una vez."""
    metrics = {
        "model":  model_name,
        "rmse":   rmse(y_true, y_pred),
        "mae":    mae(y_true, y_pred),
        "mape":   mape(y_true, y_pred),
        "smape":  smape(y_true, y_pred),
    }
    logger.info(
        f"[{model_name}] RMSE={metrics['rmse']:,.2f}  "
        f"MAE={metrics['mae']:,.2f}  "
        f"MAPE={metrics['mape']:.2f}%  "
        f"sMAPE={metrics['smape']:.2f}%"
    )
    return metrics


# ─────────────────────────────────────────────────────────────
#  Comparación de modelos
# ─────────────────────────────────────────────────────────────
def compare_models(results: list[dict], sort_by: str = "mape") -> pd.DataFrame:
    """
    Genera tabla comparativa de métricas para múltiples modelos.

    Parámetros
    ----------
    results : lista de dicts con keys model, rmse, mae, mape, smape
    sort_by : columna para ordenar (por defecto mape)
    """
    df = pd.DataFrame(results).sort_values(sort_by)
    df["meets_target"] = df["mape"] < 10.0
    return df


# ─────────────────────────────────────────────────────────────
#  Métricas de negocio
# ─────────────────────────────────────────────────────────────
def compute_business_metrics(
    df_valid: pd.DataFrame,
    monthly: pd.DataFrame
) -> dict:
    """
    Calcula KPIs de negocio del dataset Olist.

    Parámetros
    ----------
    df_valid  : Master Table filtrada (órdenes delivered + payment>0)
    monthly   : tabla mensual agregada

    Retorna
    -------
    dict con todos los KPIs de negocio.
    """
    metrics = {}

    # ── Revenue
    metrics["revenue_total"]          = float(df_valid["payment_value"].sum())
    metrics["avg_monthly_revenue"]    = float(monthly["monthly_revenue"].mean())
    metrics["median_monthly_revenue"] = float(monthly["monthly_revenue"].median())
    metrics["max_monthly_revenue"]    = float(monthly["monthly_revenue"].max())
    metrics["min_monthly_revenue"]    = float(monthly["monthly_revenue"].min())

    # ── Satisfacción
    if "review_score" in df_valid.columns:
        metrics["avg_review_score"] = float(df_valid["review_score"].mean())
        metrics["pct_5star"]        = float((df_valid["review_score"] == 5).mean() * 100)
        metrics["pct_1star"]        = float((df_valid["review_score"] == 1).mean() * 100)

    # ── Entrega
    if "delivery_days_actual" in df_valid.columns:
        metrics["avg_delivery_days"] = float(df_valid["delivery_days_actual"].mean())
    if "is_late_delivery" in df_valid.columns:
        metrics["pct_on_time"]       = float((df_valid["is_late_delivery"] == 0).mean() * 100)
    if "delivery_delay_days" in df_valid.columns:
        late = df_valid[df_valid.get("delivery_delay_days", pd.Series(dtype=float)) > 0]
        if len(late) > 0:
            metrics["avg_late_delay_days"] = float(late["delivery_delay_days"].mean())

    # ── Clientes
    if "customer_unique_id" in df_valid.columns:
        metrics["total_unique_customers"] = int(df_valid["customer_unique_id"].nunique())
        repeat = (
            df_valid.groupby("customer_unique_id")["order_id"].nunique() > 1
        ).mean() * 100
        metrics["repeat_customer_rate"] = float(repeat)

    logger.info("Métricas de negocio calculadas")
    return metrics


def format_business_report(metrics: dict) -> str:
    """Formatea el reporte de métricas de negocio como string."""
    lines = ["\n" + "="*60, " MÉTRICAS DE NEGOCIO", "="*60]
    for k, v in metrics.items():
        if isinstance(v, float):
            if "pct" in k or "rate" in k or "pct" in k:
                lines.append(f"  {k:<40s}: {v:.2f}%")
            elif "revenue" in k or "total" in k.lower():
                lines.append(f"  {k:<40s}: R$ {v:>15,.2f}")
            else:
                lines.append(f"  {k:<40s}: {v:.2f}")
        else:
            lines.append(f"  {k:<40s}: {v:,}")
    return "\n".join(lines)
