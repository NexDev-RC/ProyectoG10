"""
Utilidades compartidas: logging, config loading, paths, serialización.
"""
import os
import pickle
import joblib
import yaml
from pathlib import Path
from loguru import logger


# ── Ruta raíz del proyecto (dos niveles arriba de este archivo)
ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict:
    """Carga config.yaml desde config/ o ruta explícita."""
    if config_path is None:
        config_path = ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str, cfg: dict | None = None) -> Path:
    """Devuelve la ruta absoluta para una clave de paths del config."""
    if cfg is None:
        cfg = load_config()
    return ROOT / cfg["paths"][key]


def setup_logger(log_file: str = "pipeline.log") -> None:
    """Configura loguru con archivo rotativo."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / log_file,
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
    )


def save_model(model, name: str, cfg: dict | None = None) -> Path:
    """Serializa un modelo con joblib al directorio data/models/."""
    if cfg is None:
        cfg = load_config()
    model_dir = get_path("data_models", cfg)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Modelo guardado: {path}")
    return path


def load_model(name: str, cfg: dict | None = None):
    """Carga un modelo serializado por nombre."""
    if cfg is None:
        cfg = load_config()
    path = get_path("data_models", cfg) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)


def save_dataframe(df, name: str, cfg: dict | None = None) -> Path:
    """Guarda un DataFrame en parquet en data/processed/."""
    if cfg is None:
        cfg = load_config()
    proc_dir = get_path("data_processed", cfg)
    proc_dir.mkdir(parents=True, exist_ok=True)
    path = proc_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"DataFrame guardado: {path} ({len(df):,} filas)")
    return path


def load_dataframe(name: str, cfg: dict | None = None):
    """Carga un DataFrame desde parquet."""
    import pandas as pd
    if cfg is None:
        cfg = load_config()
    path = get_path("data_processed", cfg) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"DataFrame no encontrado: {path}")
    return pd.read_parquet(path)
