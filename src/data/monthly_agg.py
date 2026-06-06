"""
Paso 3 / 4 – Agregación mensual y construcción del target monthly_revenue.

Toma la Master Table transaccional y genera la serie temporal mensual
con todas las features agregadas (base para el modelo de forecast).
"""
import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────
#  Agregación mensual base
# ─────────────────────────────────────────────────────────────
def build_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la Master Table transaccional a nivel mensual.

    Filtra solo órdenes entregadas con payment_value > 0.
    Elimina primer y último mes (datos incompletos).

    Retorna
    -------
    DataFrame mensual con ~20 columnas base (sin lags ni rolling).
    """
    logger.info("Construyendo tabla mensual…")

    # Filtrar órdenes válidas
    df_valid = df[
        (df["order_status"] == "delivered") &
        (df["payment_value"].notna()) &
        (df["payment_value"] > 0)
    ].copy()

    n_valid = df_valid["order_id"].nunique()
    logger.info(f"Órdenes válidas (delivered + payment>0): {n_valid:,}")

    # Agregación mensual
    monthly = df_valid.groupby("year_month").agg(
        monthly_revenue           = ("payment_value",              "sum"),
        monthly_orders            = ("order_id",                   "nunique"),
        monthly_customers         = ("customer_unique_id",         "nunique"),
        monthly_items             = ("order_item_id",              "count"),
        avg_ticket                = ("payment_value",              "mean"),
        median_ticket             = ("payment_value",              "median"),
        avg_review_score          = ("review_score",               "mean"),
        avg_delivery_days         = ("delivery_days_actual",       "mean"),
        avg_delay_days            = ("delivery_delay_days",        "mean"),
        pct_late_delivery         = ("is_late_delivery",           "mean"),
        avg_freight               = ("freight_value",              "mean"),
        avg_price                 = ("price",                      "mean"),
        total_freight             = ("freight_value",              "sum"),
        pct_credit_card           = ("is_credit_card",             "mean"),
        pct_boleto                = ("is_boleto",                  "mean"),
        pct_installments          = ("payment_installments_flag",  "mean"),
        avg_installments          = ("payment_installments",       "mean"),
        unique_categories         = ("product_category_name_english", "nunique"),
        unique_sellers            = ("seller_id",                  "nunique"),
        unique_states             = ("customer_state",             "nunique"),
    ).reset_index()

    # Convertir Period → Timestamp
    monthly["ds"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.sort_values("ds").reset_index(drop=True)

    # Eliminar primer y último mes (incompletos)
    monthly = monthly.iloc[1:-1].reset_index(drop=True)

    logger.info(
        f"Serie mensual: {len(monthly)} meses  "
        f"({monthly['ds'].min().strftime('%b %Y')} → "
        f"{monthly['ds'].max().strftime('%b %Y')})"
    )
    return monthly


# ─────────────────────────────────────────────────────────────
#  Features de serie temporal (lags, rolling, cíclicas)
# ─────────────────────────────────────────────────────────────
def add_time_series_features(monthly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Añade features de lag, rolling y cíclicas a la tabla mensual.

    Parámetros
    ----------
    monthly : tabla mensual base
    cfg     : configuración del proyecto

    Retorna
    -------
    DataFrame enriquecido listo para modelo.
    """
    lag_periods    = cfg["features"]["lag_periods"]
    rolling_windows = cfg["features"]["rolling_windows"]

    # ── Lag features
    for lag in lag_periods:
        monthly[f"revenue_lag_{lag}"]  = monthly["monthly_revenue"].shift(lag)
        monthly[f"orders_lag_{lag}"]   = monthly["monthly_orders"].shift(lag)

    # ── Rolling features (shift(1) para evitar data leakage)
    for window in rolling_windows:
        base = monthly["monthly_revenue"].shift(1)
        monthly[f"revenue_rolling_mean_{window}"] = base.rolling(window).mean()
        monthly[f"revenue_rolling_std_{window}"]  = base.rolling(window).std()

    # ── Growth features
    monthly["revenue_mom_growth"] = monthly["monthly_revenue"].pct_change(1)
    monthly["revenue_yoy_growth"] = monthly["monthly_revenue"].pct_change(12)

    # ── Features temporales cíclicas
    monthly["month"]      = monthly["ds"].dt.month
    monthly["quarter"]    = monthly["ds"].dt.quarter
    monthly["month_sin"]  = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"]  = np.cos(2 * np.pi * monthly["month"] / 12)
    monthly["is_q4"]      = (monthly["quarter"] == 4).astype(int)
    monthly["is_nov_dec"] = monthly["month"].isin([11, 12]).astype(int)

    logger.info(f"Features de series temporales añadidas: {monthly.shape[1]} cols totales")
    return monthly


# ─────────────────────────────────────────────────────────────
#  Split train / val / backtest / live según config
# ─────────────────────────────────────────────────────────────
def split_data(monthly: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    Genera los splits temporales según el Excel de trabajo:

      Train    : desde inicio hasta train_end
      Val      : train_end+1 hasta val_end  (backtest interno)
      Backtest : val_end+1 hasta live_month (evaluación final)
      Live     : live_month (último mes conocido)
      Predict  : predict_month (mes futuro a predecir)

    Retorna dict con claves: train, val, backtest, live
    """
    splits = cfg["splits"]
    train_end   = pd.Period(splits["train_end"],   freq="M")
    val_end     = pd.Period(splits["val_end"],     freq="M")
    live_month  = pd.Period(splits["live_month"],  freq="M")

    ym = monthly["year_month"]

    result = {
        "train":     monthly[ym <= train_end].reset_index(drop=True),
        "val":       monthly[(ym > train_end) & (ym <= val_end)].reset_index(drop=True),
        "backtest":  monthly[(ym > val_end) & (ym <= live_month)].reset_index(drop=True),
        "live":      monthly[ym == live_month].reset_index(drop=True),
        "all":       monthly.reset_index(drop=True),
    }

    for name, df in result.items():
        logger.info(f"  Split {name:10s}: {len(df):>3} meses")

    return result


# ─────────────────────────────────────────────────────────────
#  Simulación de llegada mensual de nuevos datos (Sprint 2)
# ─────────────────────────────────────────────────────────────
def simulate_monthly_update(
    monthly: pd.DataFrame,
    new_month_data: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """
    Simula la incorporación incremental de un nuevo mes de datos.

    Pasos:
      1. Agrega new_month_data a la tabla mensual
      2. Recalcula lags y rolling features
      3. Retorna tabla actualizada lista para re-entrenamiento o predicción

    Parámetros
    ----------
    monthly        : tabla mensual histórica
    new_month_data : DataFrame con el nuevo mes (mismas columnas base)
    cfg            : configuración

    Retorna
    -------
    DataFrame mensual actualizado con las nuevas features calculadas.
    """
    logger.info("Simulando actualización mensual de datos…")
    updated = pd.concat([monthly, new_month_data], ignore_index=True)
    updated = updated.sort_values("ds").reset_index(drop=True)
    updated = add_time_series_features(updated, cfg)
    logger.info(f"Tabla actualizada: {len(updated)} meses")
    return updated
