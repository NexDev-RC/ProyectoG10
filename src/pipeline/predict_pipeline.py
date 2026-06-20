"""
Pipeline de Predicción – generación de forecast mensual.

Carga el modelo entrenado, aplica la limpieza y predice
los próximos N meses (horizonte configurado).

Uso:
    from src.pipeline.predict_pipeline import PredictPipeline
    pipe = PredictPipeline()
    forecast = pipe.run()
    print(forecast)
"""
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from src.utils.helpers import load_config, load_model, load_dataframe, setup_logger
from src.data.loader import load_raw_datasets, parse_dates
from src.data.master_table import build_master_table, add_transaction_features
from src.data.monthly_agg import build_monthly_table, add_time_series_features
from src.features.cleaning import clean_monthly_table


class PredictPipeline:
    """
    Pipeline de predicción para un nuevo mes de datos.

    Workflow:
      1. Carga el modelo y pipeline de limpieza serializados
      2. Procesa los nuevos datos del mes (o usa el histórico)
      3. Genera predicciones para los próximos N meses
      4. Retorna DataFrame con fecha, predicción e intervalo de confianza
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        setup_logger("predict_pipeline.log")

        self.forecaster     = None
        self.cleaning_pipe  = None
        self.monthly        = None
        self.selected_features: list = []

    def load_artifacts(self):
        """Carga el modelo final y el pipeline de limpieza desde disco."""
        logger.info("Cargando artefactos del modelo…")
        self.forecaster    = load_model("lgbm_forecaster",    self.cfg)
        self.cleaning_pipe = load_model("cleaning_pipeline",  self.cfg)
        self.selected_features = self.forecaster.feature_cols_
        logger.info(f"Modelo cargado con {len(self.selected_features)} features")

    def load_latest_monthly(self, data_path: str | None = None) -> pd.DataFrame:
        """
        Carga o reconstruye la tabla mensual más reciente.

        Si existe el parquet en data/processed/, lo carga directamente.
        Si no, reconstruye desde los CSVs.
        """
        try:
            monthly = load_dataframe("monthly_features", self.cfg)
            logger.info(f"Tabla mensual cargada desde caché ({len(monthly)} meses)")
            return monthly
        except FileNotFoundError:
            logger.warning("Caché no encontrado, reconstruyendo tabla mensual…")

        datasets = load_raw_datasets(self.cfg, data_path)
        datasets = parse_dates(datasets)
        df_master = build_master_table(datasets)
        df_master = add_transaction_features(df_master)
        monthly   = build_monthly_table(df_master)
        monthly   = add_time_series_features(monthly, self.cfg)
        return monthly

    def apply_cleaning(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """Aplica el pipeline de limpieza (solo transform, no fit)."""
        monthly_clean, _ = clean_monthly_table(
            monthly, self.cfg, fit=False, pipeline=self.cleaning_pipe
        )
        return monthly_clean

    def generate_forecast(
        self,
        monthly_clean: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """
        Genera predicciones de revenue para los próximos N meses.

        Parámetros
        ----------
        monthly_clean : tabla mensual limpiada
        horizon       : meses a predecir (None = usa config)

        Retorna
        -------
        DataFrame con columnas: ds, yhat, yhat_lower, yhat_upper, mape_target
        """
        if horizon is None:
            horizon = self.cfg["project"]["horizon_months"]

        # El modelo es direct multi-step: tiene un sub-modelo por horizonte y
        # solo puede predecir hasta `forecaster.horizon` meses. Si se solicita
        # más, se limita a esa capacidad (evita el desajuste de longitudes que
        # provocaba "All arrays must be of the same length").
        model_horizon = getattr(self.forecaster, "horizon", horizon)
        if horizon > model_horizon:
            logger.warning(
                f"Horizonte solicitado ({horizon}) excede la capacidad del "
                f"modelo ({model_horizon}). Se limita a {model_horizon} meses."
            )

        # Última fila de features para predecir desde ese punto
        X_pred = monthly_clean[self.selected_features].tail(1)

        # Intervalos de confianza basados en residuos in-sample (Sprint 3),
        # reemplaza el supuesto anterior de ±15% fijo.
        intervals = self.forecaster.predict_with_intervals(X_pred)
        intervals = intervals.head(horizon).reset_index(drop=True)

        # El nº de meses efectivo es el de filas realmente devueltas: garantiza
        # que `future_dates` e `intervals` tengan la misma longitud.
        eff_horizon = len(intervals)
        last_date = monthly_clean["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=eff_horizon,
            freq="MS"
        )

        forecast = pd.DataFrame({
            "ds":          future_dates,
            "yhat":        intervals["yhat"].values,
            "yhat_lower":  intervals["yhat_lower"].values,
            "yhat_upper":  intervals["yhat_upper"].values,
            "mape_target": 10.0,
        })

        logger.info(f"\nForecast para los próximos {eff_horizon} meses:")
        for _, row in forecast.iterrows():
            logger.info(
                f"  {row['ds'].strftime('%b %Y')}: "
                f"R$ {row['yhat']:>12,.2f} "
                f"[{row['yhat_lower']:>12,.2f} – {row['yhat_upper']:>12,.2f}]"
            )

        return forecast

    def run(
        self,
        data_path: str | None = None,
        horizon: int | None = None,
        new_month_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de predicción.

        Parámetros
        ----------
        data_path       : ruta alternativa a los CSVs
        horizon         : meses a predecir
        new_month_data  : datos del nuevo mes (para actualización incremental)

        Retorna
        -------
        DataFrame con el forecast.
        """
        self.load_artifacts()
        monthly = self.load_latest_monthly(data_path)

        # Actualización incremental si llegan nuevos datos
        if new_month_data is not None:
            from src.data.monthly_agg import simulate_monthly_update
            monthly = simulate_monthly_update(monthly, new_month_data, self.cfg)
            logger.info("Datos del nuevo mes incorporados al histórico")

        monthly_clean = self.apply_cleaning(monthly)
        forecast = self.generate_forecast(monthly_clean, horizon=horizon)

        return forecast
