import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import rmse, mape, mae, smape, compute_technical_metrics
from src.models.baseline import NaiveModel, MovingAverageModel, SeasonalNaiveModel, LinearTrendModel
from src.features.cleaning import OutlierClipper, NaNImputer
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
