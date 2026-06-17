"""Pruebas de los entregables de Sprint 4: SHAP y métricas de negocio/ROI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import LightGBMForecaster
from src.evaluation.shap_analysis import (
    compute_shap_values, shap_summary_df, local_contributions_df, build_explainer,
)
from src.evaluation.metrics import compute_roi_metrics


@pytest.fixture
def trained_forecaster():
    """Entrena un LightGBMForecaster pequeño sobre datos sintéticos."""
    np.random.seed(0)
    n = 36
    df = pd.DataFrame({
        "revenue_lag_1": np.random.uniform(8e5, 1.5e6, n),
        "monthly_orders": np.random.uniform(2000, 6000, n),
        "avg_ticket": np.random.uniform(120, 200, n),
    })
    y = (df["revenue_lag_1"] * 0.9 + df["monthly_orders"] * 50
         + np.random.normal(0, 1e4, n))
    feats = df.columns.tolist()
    fc = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1}, horizon=3)
    fc.fit(df, y, feature_cols=feats)
    return fc, df


@pytest.fixture
def monthly():
    dates = pd.date_range("2017-01", periods=24, freq="MS")
    return pd.DataFrame({
        "ds": dates,
        "monthly_revenue": np.random.uniform(8e5, 1.5e6, 24),
    })


@pytest.fixture
def cfg():
    return {
        "metrics": {"target_mape": 10.0},
        "business": {
            "manual_hours_per_month": 16,
            "analyst_hourly_cost_brl": 80.0,
            "planning_error_cost_rate": 0.10,
            "baseline_mape": 7.12,
        },
    }


class TestShap:
    def test_explainer_builds(self, trained_forecaster):
        fc, _ = trained_forecaster
        assert build_explainer(fc, horizon=1) is not None

    def test_shap_shape(self, trained_forecaster):
        fc, X = trained_forecaster
        sv, fnames, base = compute_shap_values(fc, X, horizon=1)
        assert sv.shape == (len(X), len(fc.feature_cols_))
        assert fnames == fc.feature_cols_
        assert isinstance(base, float)

    def test_summary_sorted_desc(self, trained_forecaster):
        fc, X = trained_forecaster
        sv, fnames, _ = compute_shap_values(fc, X, horizon=1)
        summary = shap_summary_df(sv, fnames)
        vals = summary["mean_abs_shap"].values
        assert (np.diff(vals) <= 1e-9).all()  # orden descendente
        assert (vals >= 0).all()              # |SHAP| siempre no negativo

    def test_local_contributions(self, trained_forecaster):
        fc, X = trained_forecaster
        sv, fnames, _ = compute_shap_values(fc, X, horizon=1)
        loc = local_contributions_df(sv, fnames, index=-1)
        assert set(loc.columns) == {"feature", "shap_value"}
        assert len(loc) == len(fnames)

    def test_invalid_horizon(self, trained_forecaster):
        fc, X = trained_forecaster
        with pytest.raises(ValueError):
            compute_shap_values(fc, X, horizon=99)


class TestRoi:
    def test_roi_keys(self, monthly, cfg):
        fc = pd.DataFrame({"ds": pd.date_range("2018-09", periods=3, freq="MS"),
                           "yhat": [1.2e6, 1.3e6, 1.25e6]})
        roi = compute_roi_metrics(fc, monthly, cfg, model_mape=10.14)
        for k in ("time_saving_brl_per_year", "accuracy_saving_brl_per_year",
                  "total_benefit_brl_per_year", "projected_revenue_horizon"):
            assert k in roi

    def test_time_saving_value(self, monthly, cfg):
        roi = compute_roi_metrics(None, monthly, cfg, model_mape=10.0)
        # 16 h/mes * 12 * 80 BRL = 15360
        assert roi["time_saving_brl_per_year"] == pytest.approx(15360.0)

    def test_accuracy_saving_positive_when_model_beats_baseline(self, monthly, cfg):
        roi = compute_roi_metrics(None, monthly, cfg, model_mape=5.0)
        # baseline 7.12 > 5.0 → mejora de 2.12 pts → ahorro > 0
        assert roi["mape_improvement_pts"] == pytest.approx(2.12)
        assert roi["accuracy_saving_brl_per_year"] > 0

    def test_accuracy_saving_zero_when_worse(self, monthly, cfg):
        roi = compute_roi_metrics(None, monthly, cfg, model_mape=12.0)
        assert roi["mape_improvement_pts"] == 0.0
        assert roi["accuracy_saving_brl_per_year"] == 0.0

    def test_projected_revenue(self, monthly, cfg):
        fc = pd.DataFrame({"yhat": [1.0e6, 2.0e6]})
        roi = compute_roi_metrics(fc, monthly, cfg, model_mape=10.0)
        assert roi["projected_revenue_horizon"] == pytest.approx(3.0e6)
        assert roi["forecast_horizon_months"] == 2
