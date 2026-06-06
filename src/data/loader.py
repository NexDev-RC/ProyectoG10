"""
Paso 1 – Carga de datasets Olist.

Soporta tres fuentes:
  - kaggle   : descarga automática vía API
  - local    : archivos CSV en data/raw/
  - drive    : Google Drive (uso en Colab)
"""
import os
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.helpers import load_config, get_path


# ─────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────
DATE_COLS_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ─────────────────────────────────────────────────────────────
#  Descarga desde Kaggle
# ─────────────────────────────────────────────────────────────
def download_from_kaggle(cfg: dict) -> Path:
    """Descarga el dataset Olist desde Kaggle si aún no existe."""
    raw_dir = get_path("data_raw", cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Verificar si ya existen los CSV
    expected = cfg["dataset"]["files"]["orders"]
    if (raw_dir / expected).exists():
        logger.info("Dataset ya descargado, usando caché local.")
        return raw_dir

    kaggle_dataset = cfg["dataset"]["kaggle_dataset"]
    logger.info(f"Descargando {kaggle_dataset} desde Kaggle…")

    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_dataset,
         "--unzip", "-p", str(raw_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error Kaggle: {result.stderr}")

    logger.info(f"Dataset descargado en {raw_dir}")
    return raw_dir


# ─────────────────────────────────────────────────────────────
#  Carga de CSVs
# ─────────────────────────────────────────────────────────────
def load_raw_datasets(cfg: dict | None = None, data_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Carga todos los CSV de Olist en un diccionario.

    Parámetros
    ----------
    cfg       : configuración del proyecto (carga automática si None)
    data_path : ruta explícita a la carpeta de CSVs (override)

    Retorna
    -------
    dict con claves: orders, order_items, order_payments, order_reviews,
                     customers, products, sellers, category_trans, geolocation
    """
    if cfg is None:
        cfg = load_config()

    if data_path is None:
        source = cfg["dataset"].get("source", "local")
        if source == "kaggle":
            data_path = download_from_kaggle(cfg)
        else:
            data_path = get_path("data_raw", cfg)

    data_path = Path(data_path)
    files = cfg["dataset"]["files"]

    datasets: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        filepath = data_path / filename
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            continue
        df = pd.read_csv(filepath)
        datasets[key] = df
        logger.info(f"  {key:20s}: {df.shape[0]:>8,} filas x {df.shape[1]:>3} cols")

    logger.info(f"Total datasets cargados: {len(datasets)}")
    return datasets


# ─────────────────────────────────────────────────────────────
#  Conversión de fechas
# ─────────────────────────────────────────────────────────────
def parse_dates(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convierte columnas de fecha a datetime."""
    if "orders" in datasets:
        for col in DATE_COLS_ORDERS:
            if col in datasets["orders"].columns:
                datasets["orders"][col] = pd.to_datetime(
                    datasets["orders"][col], errors="coerce"
                )

    if "order_items" in datasets:
        datasets["order_items"]["shipping_limit_date"] = pd.to_datetime(
            datasets["order_items"]["shipping_limit_date"], errors="coerce"
        )

    rng = datasets["orders"]["order_purchase_timestamp"]
    logger.info(
        f"Rango temporal: {rng.min().date()} → {rng.max().date()} "
        f"({rng.dt.to_period('M').nunique()} meses)"
    )
    return datasets


# ─────────────────────────────────────────────────────────────
#  Resumen de calidad
# ─────────────────────────────────────────────────────────────
def data_quality_report(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera reporte de calidad por dataset."""
    rows = []
    for name, df in datasets.items():
        rows.append({
            "dataset":       name,
            "rows":          df.shape[0],
            "cols":          df.shape[1],
            "null_pct":      round(df.isnull().mean().mean() * 100, 2),
            "dup_rows":      df.duplicated().sum(),
        })
    report = pd.DataFrame(rows)
    logger.info("\n" + report.to_string(index=False))
    return report
