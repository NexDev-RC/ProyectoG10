"""Tests de las mejoras del Sprint 3 (flujo de ejemplo adaptado a series temporales)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.sklearn_forecaster import SklearnForecaster
from src.features.selection_table import infer_domain, build_selection_table
from src.evaluation.gap_report import train_backtest_gap, gap_summary, selection_cost


@pytest.fixture
def Xy():
    np.random.seed(42)
    n = 18
    X = pd.DataFrame({
        "monthly_orders": np.linspace(2000, 6000, n) + np.random.normal(0, 100, n),
        "month":          (np.arange(n) % 12) + 1.0,
        "quarter":        ((np.arange(n) % 12) // 3) + 1.0,
    })
    y = pd.Series(300.0 * X["monthly_orders"] + np.random.normal(0, 5000, n),
                  name="monthly_revenue")
    return X, y


class TestSklearnForecaster:
    def test_fit_predict_shapes(self, Xy):
        X, y = Xy
        fc = SklearnForecaster(Ridge(alpha=1.0), name="Ridge", horizon=3)
        fc.fit(X, y)
        preds = fc.predict(X.tail(1))
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))

    def test_one_model_per_horizon(self, Xy):
        X, y = Xy
        fc = SklearnForecaster(Ridge(), horizon=3).fit(X, y)
        assert set(fc.models_.keys()) == {1, 2, 3}
        assert set(fc.resid_std_.keys()) == {1, 2, 3}

    def test_feature_importance_coef(self, Xy):
        X, y = Xy
        fc = SklearnForecaster(Ridge(), horizon=2).fit(X, y)
        fi = fc.feature_importance()
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == X.shape[1]


class TestSelectionTable:
    def test_infer_domain_exact_beats_prefix(self):
        # 'monthly_orders' empieza con 'month' pero su dominio correcto es Órdenes
        assert infer_domain("monthly_orders") == "Órdenes"
        assert infer_domain("month") == "Calendario / Estacionalidad"
        assert infer_domain("month_sin") == "Calendario / Estacionalidad"
        assert infer_domain("monthly_revenue") == "Target"
        assert infer_domain("revenue_lag_12") == "Historia del Target (lags)"
        assert infer_domain("avg_delay_days") == "Entregas"
        assert infer_domain("columna_inventada") == "Otros"

    def test_build_table_flags_and_pct(self):
        cols = ["monthly_revenue", "monthly_orders", "month", "avg_delay_days"]
        selected = ["monthly_orders", "month"]
        uni = pd.DataFrame(
            {"mi_score": [0.8, 0.2, 0.1]},
            index=["monthly_orders", "month", "avg_delay_days"],
        )
        fi = pd.DataFrame({"feature": ["monthly_orders", "month"], "importance": [30, 10]})
        table = build_selection_table(
            all_columns=cols,
            selected_features=selected,
            steps_report={"4_univariate": {"report": uni}},
            final_importance=fi,
        )
        assert len(table) == 4
        assert table["flagSelected"].sum() == 2
        # importancia_seleccion suma 100% entre seleccionadas
        assert table["importancia_seleccion"].sum() == pytest.approx(100.0, abs=0.1)
        # el target no está seleccionado
        assert table.loc[table["Variable"] == "monthly_revenue", "flagSelected"].iloc[0] == 0


class TestGapReport:
    def test_gap_keys_and_ratio(self, Xy):
        X, y = Xy
        fc = SklearnForecaster(Ridge(), horizon=3).fit(X, y)
        y_test = np.array([2.0e6, 2.1e6, 2.2e6])
        y_pred = y_test * 1.05
        gap = train_backtest_gap(fc, X, y, y_test, y_pred, "Ridge")
        for k in ["model", "rmse_train", "rmse_backtest", "gap_rmse_ratio", "mape_backtest"]:
            assert k in gap
        assert gap["gap_rmse_ratio"] > 0
        assert gap["mape_backtest"] == pytest.approx(5.0, abs=0.2)

    def test_gap_summary_sorted(self):
        gaps = [
            {"model": "A", "rmse_train": 10, "rmse_backtest": 100, "gap_rmse_ratio": 10, "mape_backtest": 9},
            {"model": "B", "rmse_train": 50, "rmse_backtest": 60,  "gap_rmse_ratio": 1.2, "mape_backtest": 5},
        ]
        df = gap_summary(gaps)
        assert df.iloc[0]["model"] == "B"  # mejor backtest primero

    def test_selection_cost_shape(self):
        df = selection_cost(
            {"rmse": 100.0, "mape": 8.0},
            {"rmse": 90.0, "mape": 7.5},
            n_all=41, n_selected=5,
        )
        assert len(df) == 2
        assert df.iloc[1]["n_features"] == 5
