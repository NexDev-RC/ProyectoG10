import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import rmse, mape, mae, smape, compute_technical_metrics
from src.models.baseline import NaiveModel, MovingAverageModel, SeasonalNaiveModel, LinearTrendModel
from src.features.cleaning import (
    OutlierClipper, NaNImputer, RareCategoryGrouper,
    CategoricalEncoder, FeatureScaler, build_cleaning_pipeline, clean_monthly_table,
)
from src.features.selection import missing_selection, correlation_selection, calculate_psi

@pytest.fixture
def series24():
    np.random.seed(42)
    t = np.linspace(800000, 1500000, 24)
    s = 100000 * np.sin(2 * np.pi * np.arange(24) / 12)
    return t + s + np.random.normal(0, 30000, 24)

@pytest.fixture
def monthly(series24):
    dates = pd.date_range("2017-01", periods=24, freq="MS")
    return pd.DataFrame({
        "ds": dates,
        "monthly_revenue": series24,
        "monthly_orders": np.random.randint(2000,6000,24).astype(float),
        "avg_ticket": np.random.uniform(120,200,24),
        "feature_noise": np.random.normal(0,1,24),
    })

class TestMetrics:
    def test_rmse_zero(self):
        y = np.array([100.0, 200.0])
        assert rmse(y, y) == pytest.approx(0.0)
    def test_rmse_known(self):
        assert rmse(np.array([100.0,200.0]), np.array([110.0,190.0])) == pytest.approx(10.0)
    def test_mape_perfect(self):
        y = np.array([100.0, 200.0])
        assert mape(y, y) == pytest.approx(0.0)
    def test_mape_10pct(self):
        assert mape(np.array([100.0]), np.array([110.0])) == pytest.approx(10.0)
    def test_mae_known(self):
        assert mae(np.array([100.0,200.0]), np.array([110.0,210.0])) == pytest.approx(10.0)
    def test_smape_nonneg(self, series24):
        y = series24[:10]
        assert smape(y, y+1000) >= 0
    def test_metrics_dict_keys(self, series24):
        m = compute_technical_metrics(series24[:10]+1000, series24[:10], "M")
        assert all(k in m for k in ["model","rmse","mape","mae","smape"])

class TestBaseline:
    def test_naive(self, series24):
        p = NaiveModel().fit(series24[:18]).predict(3)
        assert len(p) == 3 and all(v == series24[17] for v in p)
    def test_ma3(self, series24):
        p = MovingAverageModel(3).fit(series24[:18]).predict(3)
        assert all(v == pytest.approx(float(np.mean(series24[15:18]))) for v in p)
    def test_seasonal_len(self, series24):
        assert len(SeasonalNaiveModel().fit(series24[:18]).predict(3)) == 3
    def test_linear_increasing(self):
        t = np.linspace(100000, 500000, 20)
        assert LinearTrendModel().fit(t).predict(1)[0] > t[-1]

class TestCleaning:
    def test_clipper_reduces_range(self):
        vals = list(range(1,101)) + [9999.0, -9999.0]
        d = pd.DataFrame({"x": vals})
        r = OutlierClipper(0.05, 0.95).fit_transform(d)
        assert r["x"].max() < 9999.0
        assert r["x"].min() > -9999.0
    def test_nan_median(self):
        d = pd.DataFrame({"x": [1.0,2.0,np.nan,4.0,5.0]})
        r = NaNImputer("median").fit_transform(d)
        assert r["x"].isnull().sum() == 0
        assert r["x"].iloc[2] == pytest.approx(3.0)
    def test_nan_ffill(self):
        d = pd.DataFrame({"x": [1.0,2.0,np.nan,np.nan,5.0]})
        r = NaNImputer("forward_fill").fit_transform(d)
        assert r["x"].isnull().sum() == 0
    def test_scaler_standard_mean_zero(self):
        d = pd.DataFrame({"x": np.arange(1.0, 21.0), "flag": [0,1]*10})
        r = FeatureScaler("standard").fit_transform(d)
        assert r["x"].mean() == pytest.approx(0.0, abs=1e-9)
        assert r["x"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)
    def test_scaler_excludes_binary(self):
        d = pd.DataFrame({"x": np.arange(1.0, 21.0), "flag": [0,1]*10})
        r = FeatureScaler("standard", exclude_binary=True).fit_transform(d)
        assert (r["flag"] == d["flag"]).all()
    def test_scaler_none_passthrough(self):
        d = pd.DataFrame({"x": np.arange(1.0, 21.0)})
        r = FeatureScaler("none").fit_transform(d)
        pd.testing.assert_frame_equal(r, d)
    def test_categorical_encoder_one_hot(self):
        d = pd.DataFrame({"cat": ["a","b","a","c"], "x":[1.0,2.0,3.0,4.0]})
        enc = CategoricalEncoder().fit(d)
        r = enc.transform(d)
        assert "cat" not in r.columns
        assert set(["cat__a","cat__b","cat__c"]).issubset(r.columns)
        assert r.loc[0, "cat__a"] == 1 and r.loc[0, "cat__b"] == 0
    def test_categorical_encoder_passthrough_numeric(self):
        d = pd.DataFrame({"x":[1.0,2.0,3.0]})
        r = CategoricalEncoder().fit_transform(d)
        pd.testing.assert_frame_equal(r, d)
    def test_cleaning_pipeline_full(self):
        cfg = {"cleaning": {
            "clip_quantile_low": 0.01, "clip_quantile_high": 0.99,
            "nan_fill_strategy": "median", "group_rare_threshold": 0.02,
            "scaling_method": "standard",
        }}
        d = pd.DataFrame({
            "year_month": pd.period_range("2018-01", periods=10, freq="M"),
            "ds": pd.date_range("2018-01-01", periods=10, freq="MS"),
            "monthly_revenue": np.linspace(1000, 2000, 10),
            "x1": np.linspace(1, 10, 10),
            "x2": [1,1,1,1,0,0,0,0,1,1],
        })
        out, pipe = clean_monthly_table(d, cfg, fit=True)
        assert "monthly_revenue" in out.columns and "ds" in out.columns
        assert out["x1"].mean() == pytest.approx(0.0, abs=1e-9)
        # transform-only (predict path) reusa el pipeline ajustado
        out2, _ = clean_monthly_table(d, cfg, fit=False, pipeline=pipe)
        assert list(out2.columns) == list(out.columns)

class TestSelection:
    def test_missing_drops_high_null(self):
        d = pd.DataFrame({"ok":[1.0,2.0,3.0,4.0,5.0],"bad":[np.nan,np.nan,np.nan,1.0,np.nan]})
        sel, _ = missing_selection(d, 0.15)
        assert "ok" in sel and "bad" not in sel
    def test_corr_drops_redundant(self):
        np.random.seed(0)
        x = np.random.randn(50)
        d = pd.DataFrame({"a":x,"b":x+np.random.randn(50)*0.01,"c":np.random.randn(50)})
        sel, _ = correlation_selection(d, 0.95)
        assert len(sel) == 2
    def test_psi_stable(self):
        np.random.seed(0)
        assert calculate_psi(pd.Series(np.random.normal(0,1,500)), pd.Series(np.random.normal(0,1,500))) < 0.10
    def test_psi_unstable(self):
        assert calculate_psi(pd.Series(np.random.normal(0,1,500)), pd.Series(np.random.normal(10,1,500))) > 0.20

class TestIncrementalUpdate:
    def test_lgbm_forecaster_update_warm_start(self):
        from src.models.forecaster import LightGBMForecaster

        np.random.seed(0)
        n = 30
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = pd.Series(X["f1"] * 2 + X["f2"] + np.random.randn(n) * 0.1 + 100)

        params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1,
            "n_estimators": 20, "min_child_samples": 1, "min_data_in_leaf": 1,
            "random_state": 42,
        }
        fc = LightGBMForecaster(params=params, horizon=1)
        fc.fit(X, y, feature_cols=["f1", "f2"])
        pred_before = fc.predict(X)
        trees_before = fc.models_[1].booster_.num_trees()

        # Ventana reciente con el/los nuevo(s) mes(es): LightGBM necesita
        # >= 2 observaciones para evaluar particiones nuevas durante el
        # continue-training (warm start).
        X_new = pd.DataFrame({"f1": [0.3, 0.5], "f2": [0.1, -0.5]})
        y_new = pd.Series([100.7, 101.0])
        fc.update(X_new, y_new, n_extra_trees=5)
        pred_after = fc.predict(X)
        trees_after = fc.models_[1].booster_.num_trees()

        assert pred_before.shape == pred_after.shape
        # El warm start continua el boosting: el total de arboles crece
        assert trees_after > trees_before

class TestMonthly:
    def test_lag_no_leakage(self, monthly):
        from src.data.monthly_agg import add_time_series_features
        r = add_time_series_features(monthly.copy(), {"features":{"lag_periods":[1],"rolling_windows":[3]}})
        for i in range(1, len(r)):
            assert r["revenue_lag_1"].iloc[i] == pytest.approx(r["monthly_revenue"].iloc[i-1])
    def test_more_features_added(self, monthly):
        from src.data.monthly_agg import add_time_series_features
        before = len(monthly.columns)
        r = add_time_series_features(monthly.copy(), {"features":{"lag_periods":[1,2,3],"rolling_windows":[3,6]}})
        assert len(r.columns) > before
