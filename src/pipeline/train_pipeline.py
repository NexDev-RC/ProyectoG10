"""
Pipeline de Entrenamiento Completo – Sprint 1 + Sprint 2.

Orquesta todos los pasos del Excel:
  Paso 1 : EDA / Carga de datos
  Paso 2 : Definición de splits temporales
  Paso 3 : Feature engineering
  Paso 4 : Master Table mensual
  Paso 5 : Limpieza (clip, NaN, agrupamiento)
  Paso 6 : Selección de variables
  Paso 7 : Hiperparametrización (Optuna)
  Paso 8 : Entrenamiento del modelo final

Uso:
    from src.pipeline.train_pipeline import TrainPipeline
    pipe = TrainPipeline()
    pipe.run()
"""
import mlflow
import pandas as pd
from loguru import logger
from pathlib import Path

from src.utils.helpers import load_config, save_model, save_dataframe, setup_logger
from src.data.loader import load_raw_datasets, parse_dates, data_quality_report
from src.data.master_table import build_master_table, add_transaction_features
from src.data.monthly_agg import build_monthly_table, add_time_series_features, split_data
from src.features.cleaning import clean_monthly_table
from src.features.selection import run_feature_selection
from src.models.baseline import evaluate_baselines
from src.models.trainer import tune_lightgbm, train_final_model
from src.models.forecaster import ProphetForecaster, SARIMAForecaster
from src.evaluation.metrics import (
    compute_technical_metrics,
    compute_business_metrics,
    compare_models,
    format_business_report,
)


class TrainPipeline:
    """
    Pipeline de entrenamiento orquestado.

    Atributos públicos después de run():
      self.monthly        : tabla mensual completa
      self.splits         : dict de splits
      self.selected_features : lista de features seleccionadas
      self.forecaster     : modelo final entrenado
      self.cleaning_pipe  : pipeline de limpieza ajustado
      self.metrics_report : dict de métricas finales
    """

    def __init__(self, cfg: dict | None = None, data_path: str | None = None):
        self.cfg       = cfg or load_config()
        self.data_path = data_path
        setup_logger("train_pipeline.log")

        # Estado interno
        self.datasets:           dict = {}
        self.df_master:          pd.DataFrame | None = None
        self.monthly:            pd.DataFrame | None = None
        self.splits:             dict = {}
        self.cleaning_pipe:      object = None
        self.selected_features:  list = []
        self.forecaster:         object = None
        self.metrics_report:     dict = {}

    # ──────────────────────────────────────────────
    #  Paso 1+2: Carga y preparación de datos
    # ──────────────────────────────────────────────
    def step_load_data(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 1+2: CARGA DE DATOS")
        logger.info("═"*60)

        self.datasets = load_raw_datasets(self.cfg, self.data_path)
        self.datasets = parse_dates(self.datasets)
        data_quality_report(self.datasets)

    # ──────────────────────────────────────────────
    #  Paso 3+4: Master Table + Agregación mensual
    # ──────────────────────────────────────────────
    def step_build_features(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 3+4: MASTER TABLE Y FEATURES MENSUALES")
        logger.info("═"*60)

        # Master Table transaccional
        self.df_master = build_master_table(self.datasets)
        self.df_master = add_transaction_features(self.df_master)

        # Tabla mensual con target
        self.monthly = build_monthly_table(self.df_master)
        self.monthly = add_time_series_features(self.monthly, self.cfg)

        logger.info(f"Monthly table: {self.monthly.shape[1]} features, {len(self.monthly)} meses")
        save_dataframe(self.monthly, "monthly_features", self.cfg)

    # ──────────────────────────────────────────────
    #  Paso 2: Splits
    # ──────────────────────────────────────────────
    def step_split(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 2: SPLITS TEMPORALES")
        logger.info("═"*60)
        self.splits = split_data(self.monthly, self.cfg)

    # ──────────────────────────────────────────────
    #  Baseline (Sprint 1)
    # ──────────────────────────────────────────────
    def step_baseline(self):
        logger.info("\n" + "═"*60)
        logger.info("BASELINE MODELS (Sprint 1)")
        logger.info("═"*60)

        train = self.splits["train"]
        test  = pd.concat([self.splits["val"], self.splits["backtest"]])
        n_test = min(3, len(test))

        baseline_results = evaluate_baselines(train, test, n_test=n_test)
        self.metrics_report["baseline"] = baseline_results.to_dict(orient="records")
        logger.info("\n" + baseline_results.to_string(index=False))

    # ──────────────────────────────────────────────
    #  Paso 5: Limpieza
    # ──────────────────────────────────────────────
    def step_clean(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 5: LIMPIEZA DE VARIABLES")
        logger.info("═"*60)

        self.monthly, self.cleaning_pipe = clean_monthly_table(
            self.monthly, self.cfg, fit=True
        )
        # Re-split con datos limpios
        self.splits = split_data(self.monthly, self.cfg)
        save_model(self.cleaning_pipe, "cleaning_pipeline", self.cfg)

    # ──────────────────────────────────────────────
    #  Paso 6: Selección de variables
    # ──────────────────────────────────────────────
    def step_select_features(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 6: SELECCIÓN DE VARIABLES")
        logger.info("═"*60)

        train = self.splits["train"]
        val   = self.splits["val"]

        if len(val) == 0:
            val = train.tail(3)

        selection_result = run_feature_selection(
            monthly_train=train,
            monthly_val=val,
            target_col=self.cfg["project"]["target"],
            cfg=self.cfg,
        )
        self.selected_features = selection_result["selected_features"]
        self.metrics_report["feature_selection"] = {
            "n_initial": len([c for c in train.columns
                              if c not in {"year_month", "ds", "monthly_revenue"}]),
            "n_selected": len(self.selected_features),
            "selected":   self.selected_features,
        }

        logger.info(f"Features finales: {self.selected_features}")

    # ──────────────────────────────────────────────
    #  Paso 7: Hiperparametrización (Optuna)
    # ──────────────────────────────────────────────
    def step_tune(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 7: HIPERPARAMETRIZACIÓN (OPTUNA)")
        logger.info("═"*60)

        train  = self.splits["train"]
        target = self.cfg["project"]["target"]

        X_train = train[self.selected_features]
        y_train = train[target]

        tuning_result = tune_lightgbm(X_train, y_train, self.selected_features, self.cfg)
        self.best_params = tuning_result["best_params"]
        self.metrics_report["optuna"] = {
            "best_value":  tuning_result["best_value"],
            "best_params": self.best_params,
        }

    # ──────────────────────────────────────────────
    #  Paso 8: Entrenamiento modelo final
    # ──────────────────────────────────────────────
    def step_train_final(self):
        logger.info("\n" + "═"*60)
        logger.info("PASO 8: ENTRENAMIENTO MODELO FINAL")
        logger.info("═"*60)

        all_data = self.splits["all"]
        target   = self.cfg["project"]["target"]
        horizon  = self.cfg["project"]["horizon_months"]

        X_all = all_data[self.selected_features]
        y_all = all_data[target]

        self.forecaster = train_final_model(
            X_all, y_all, self.best_params, self.selected_features, horizon=horizon
        )

        # Evaluación en backtest
        backtest = self.splits["backtest"]
        if len(backtest) > 0:
            X_bt = backtest[self.selected_features]
            y_bt = backtest[target].values
            y_pred = self.forecaster.predict(X_bt)[:len(y_bt)]

            final_metrics = compute_technical_metrics(y_bt, y_pred, "LightGBM_final")
            self.metrics_report["final_model"] = final_metrics
            logger.info(f"MAPE final: {final_metrics['mape']:.2f}%")

            if final_metrics["mape"] < 10.0:
                logger.info(" OBJETIVO ALCANZADO: MAPE < 10%")
            else:
                logger.warning(f" MAPE={final_metrics['mape']:.2f}% > 10% – revisar features/modelo")

        save_model(self.forecaster, "lgbm_forecaster", self.cfg)

    # ──────────────────────────────────────────────
    #  Sprint 2: Actualización incremental mensual
    # ──────────────────────────────────────────────
    def step_incremental_update(self, new_month_data: pd.DataFrame, update_window: int = 3):
        """
        Simula la llegada de un nuevo mes de datos y actualiza el modelo
        de forma incremental (warm start), sin reentrenar desde cero.

        Flujo:
          1. Incorpora el nuevo mes y recalcula lags/rolling/cíclicas
             (simulate_monthly_update).
          2. Re-aplica el pipeline de limpieza ya ajustado (transform,
             sin refit) → limpieza, encoding y escalado consistentes.
          3. Actualiza el modelo LightGBM con una ventana reciente de
             observaciones (forecaster.update, warm start).
          4. Recalcula métricas técnicas sobre el backtest actualizado.

        Parámetros
        ----------
        new_month_data : DataFrame con el nuevo mes (mismas columnas base
                          que produce build_monthly_table, p.ej. salida de
                          add_time_series_features para ese mes o un row
                          equivalente).
        update_window  : número de meses recientes (incluyendo el nuevo)
                          usados para la actualización incremental. LightGBM
                          no puede continuar el boosting con un único punto
                          (no hay forma de evaluar una partición), por lo
                          que se usa una pequeña ventana deslizante de meses
                          recientes para el warm start.

        Retorna
        -------
        dict con las métricas técnicas post-actualización.
        """
        from src.data.monthly_agg import split_data, simulate_monthly_update

        logger.info("\n" + "═"*60)
        logger.info("SPRINT 2: ACTUALIZACIÓN INCREMENTAL MENSUAL")
        logger.info("═"*60)

        if self.cleaning_pipe is None or self.forecaster is None:
            raise RuntimeError(
                "Se requiere un pipeline entrenado (step_clean + step_train_final) "
                "antes de simular una actualización incremental."
            )

        target = self.cfg["project"]["target"]

        # 1. Incorporar el nuevo mes y recalcular features temporales
        self.monthly = simulate_monthly_update(self.monthly, new_month_data, self.cfg)

        # 2. Re-aplicar limpieza (encoding + escalado) ya ajustada, sin refit
        monthly_clean, _ = clean_monthly_table(
            self.monthly, self.cfg, fit=False, pipeline=self.cleaning_pipe
        )
        self.monthly = monthly_clean
        self.splits = split_data(self.monthly, self.cfg)

        # 3. Actualización incremental (warm start) con una ventana reciente
        #    (incluye el nuevo mes); LightGBM requiere >= 2 observaciones
        #    para poder añadir nuevos splits durante el continue-training.
        window = monthly_clean.tail(max(update_window, 2))
        X_new = window[self.selected_features]
        y_new = window[target]
        self.forecaster.update(X_new, y_new)
        save_model(self.forecaster, "lgbm_forecaster", self.cfg)

        # 4. Métricas técnicas post-actualización sobre el backtest disponible
        backtest = self.splits["backtest"]
        updated_metrics = {}
        if len(backtest) > 0:
            X_bt = backtest[self.selected_features]
            y_bt = backtest[target].values
            y_pred = self.forecaster.predict(X_bt)[:len(y_bt)]
            updated_metrics = compute_technical_metrics(y_bt, y_pred, "LightGBM_incremental")
            logger.info(f"MAPE post-actualización: {updated_metrics['mape']:.2f}%")

        self.metrics_report["incremental_update"] = updated_metrics
        logger.info("Actualización incremental completada.")
        return updated_metrics

    # ──────────────────────────────────────────────
    #  Métricas de negocio
    # ──────────────────────────────────────────────
    def step_business_metrics(self):
        logger.info("\n" + "═"*60)
        logger.info("MÉTRICAS DE NEGOCIO")
        logger.info("═"*60)

        # Filtrar datos válidos para métricas de negocio
        df_valid = self.df_master[
            (self.df_master["order_status"] == "delivered") &
            (self.df_master["payment_value"] > 0)
        ]
        biz_metrics = compute_business_metrics(df_valid, self.monthly)
        self.metrics_report["business"] = biz_metrics
        print(format_business_report(biz_metrics))

    # ──────────────────────────────────────────────
    #  Pipeline completo
    # ──────────────────────────────────────────────
    def run(self, tune: bool = True, track_mlflow: bool = False):
        """
        Ejecuta el pipeline completo de extremo a extremo.

        Parámetros
        ----------
        tune         : si True, ejecuta Optuna (puede tardar varios minutos)
        track_mlflow : si True, registra métricas en MLflow
        """
        if track_mlflow:
            mlflow.set_experiment(self.cfg["project"]["name"])
            mlflow.start_run()

        try:
            self.step_load_data()
            self.step_build_features()
            self.step_split()
            self.step_baseline()
            self.step_clean()
            self.step_select_features()

            if tune:
                self.step_tune()
            else:
                # Hiperparámetros por defecto si no se tunea
                self.best_params = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": 5,
                }

            self.step_train_final()
            self.step_business_metrics()

            if track_mlflow and "final_model" in self.metrics_report:
                fm = self.metrics_report["final_model"]
                mlflow.log_metrics({
                    "rmse": fm["rmse"],
                    "mape": fm["mape"],
                    "mae":  fm["mae"],
                })
                mlflow.log_params(self.best_params)

            logger.info("\n Pipeline de entrenamiento completado exitosamente.")
            return self

        finally:
            if track_mlflow:
                mlflow.end_run()
