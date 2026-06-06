"""
Paso 4 – Construcción de la Master Table transaccional.

Une todos los datasets de Olist en una tabla plana y genera
las 60 features base (bloque transaccional). La agregación
mensual se hace en monthly_agg.py.
"""
import numpy as np
import pandas as pd
from loguru import logger


def build_master_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye la Master Table uniendo todas las tablas de Olist.

    Retorna
    -------
    DataFrame con ~119k filas y ~30 columnas base antes de feature engineering.
    """
    logger.info("Construyendo Master Table…")

    df_orders         = datasets["orders"]
    df_order_items    = datasets["order_items"]
    df_order_payments = datasets["order_payments"]
    df_order_reviews  = datasets["order_reviews"]
    df_customers      = datasets["customers"]
    df_products       = datasets["products"]
    df_sellers        = datasets["sellers"]
    df_category_trans = datasets["category_trans"]

    # ── Traducción de categorías al inglés
    df_products = df_products.merge(df_category_trans, on="product_category_name", how="left")

    # ── Merge principal
    df = (
        df_orders
        .merge(df_order_items, on="order_id", how="left")
        .merge(df_order_payments, on="order_id", how="left")
        .merge(
            df_order_reviews[["order_id", "review_score", "review_comment_title"]],
            on="order_id", how="left"
        )
        .merge(
            df_customers[["customer_id", "customer_unique_id",
                           "customer_city", "customer_state"]],
            on="customer_id", how="left"
        )
        .merge(
            df_products[["product_id", "product_category_name_english",
                          "product_weight_g", "product_length_cm",
                          "product_height_cm", "product_width_cm"]],
            on="product_id", how="left"
        )
        .merge(
            df_sellers[["seller_id", "seller_city", "seller_state"]],
            on="seller_id", how="left"
        )
    )

    logger.info(f"Master Table base: {df.shape[0]:,} filas x {df.shape[1]} cols")
    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 3 – Feature engineering transaccional (pre-agregación).

    Añade ~25 features a nivel de transacción.
    """
    logger.info("Generando features transaccionales…")

    # ── BLOQUE 1: Temporales
    ts = df["order_purchase_timestamp"]
    df["order_year"]        = ts.dt.year
    df["order_month"]       = ts.dt.month
    df["order_quarter"]     = ts.dt.quarter
    df["order_week"]        = ts.dt.isocalendar().week.astype("Int64")
    df["order_dayofweek"]   = ts.dt.dayofweek
    df["order_dayofmonth"]  = ts.dt.day
    df["order_hour"]        = ts.dt.hour
    df["is_weekend"]        = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_end_of_month"]   = (ts.dt.day >= 25).astype(int)
    df["year_month"]        = ts.dt.to_period("M")

    # ── BLOQUE 2: Entrega
    df["delivery_days_actual"] = (
        df["order_delivered_customer_date"] - ts
    ).dt.days

    df["delivery_days_estimated"] = (
        df["order_estimated_delivery_date"] - ts
    ).dt.days

    df["delivery_delay_days"] = (
        df["delivery_days_actual"] - df["delivery_days_estimated"]
    )

    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype(int)

    df["carrier_days"] = (
        df["order_delivered_carrier_date"] - ts
    ).dt.days

    # ── BLOQUE 3: Pago
    df["is_credit_card"]  = (df["payment_type"] == "credit_card").astype(int)
    df["is_boleto"]       = (df["payment_type"] == "boleto").astype(int)
    df["is_voucher"]      = (df["payment_type"] == "voucher").astype(int)
    df["is_debit_card"]   = (df["payment_type"] == "debit_card").astype(int)
    df["payment_installments_flag"] = (df["payment_installments"] > 1).astype(int)

    # ── BLOQUE 4: Producto
    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    )
    df["freight_ratio"] = df["freight_value"] / (
        df["price"] + df["freight_value"] + 1e-9
    )

    # ── BLOQUE 5: Estado de orden
    df["is_delivered"] = (df["order_status"] == "delivered").astype(int)
    df["is_canceled"]  = (df["order_status"] == "canceled").astype(int)

    logger.info(f"Features transaccionales añadidas: {df.shape[1]} cols totales")
    return df
