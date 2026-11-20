"""Tests for Phase 3 demand modeling and elasticity module."""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.demand.demand_model import (
    DEFAULT_DEMAND_FEATURES,
    DemandModel,
    fit_linear_approximation,
    load_demand_model,
    save_demand_model,
)
from src.demand.elasticity import (
    OBSERVATIONAL_CAVEAT,
    compute_per_item_elasticities,
    estimate_log_log_elasticity,
)
from src.demand.forecasting import (
    SeasonalNaiveForecaster,
    safe_wmape,
    train_and_evaluate_models,
)
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def demand_test_df() -> pd.DataFrame:
    """Load or construct sample time series dataset for demand testing."""
    path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        # Take a subset of 2 items and 2 stores for quick test execution
        sub = df[df["item_id"].isin(["FOODS_3_090", "FOODS_3_202"])].copy()
        return sub

    # Fallback synthetic fixture
    dates = pd.date_range("2015-01-01", periods=120, freq="D")
    records = []
    for d in dates:
        p = 2.0 + 0.5 * (d.day % 4 == 0)
        q = max(0.0, 50.0 - 15.0 * p + np.random.normal(0, 2))
        records.append({
            "item_id": "FOODS_3_090",
            "store_id": "CA_1",
            "date": d,
            "wm_yr_wk": 11501 + (d.day // 7),
            "sell_price": p,
            "sales": q,
            "log_price": float(np.log(p)),
            "log_sales": float(np.log1p(q)),
            "day_of_week": d.dayofweek,
            "month": d.month,
            "has_event": 0,
            "snap_CA": 0,
            "lag_sales_1": q,
            "lag_sales_7": q,
            "lag_sales_14": q,
            "lag_sales_28": q,
            "rolling_mean_7": q,
            "rolling_std_7": 1.0,
            "rolling_mean_28": q,
            "rolling_std_28": 1.0,
            "price_change_pct": 0.0,
            "price_rel_mean": 1.0,
            "is_discounted": 0,
            "is_weekend": int(d.dayofweek in [5, 6]),
            "day_of_month": d.day,
        })
    return pd.DataFrame(records)


def test_seasonal_naive_forecaster() -> None:
    """Verify SeasonalNaiveForecaster extracts lag-7 values correctly."""
    forecaster = SeasonalNaiveForecaster(lag_col="lag_sales_7")
    df = pd.DataFrame({"lag_sales_7": [10.0, 20.0, 0.0, np.nan]})
    preds = forecaster.predict(df)
    assert len(preds) == 4
    assert preds[0] == 10.0
    assert preds[1] == 20.0
    assert preds[2] == 0.0
    assert preds[3] == 0.0  # NaN filled with 0.0


def test_safe_wmape_metric() -> None:
    """Verify safe WMAPE handles zeros without ZeroDivisionError."""
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([2.0, 8.0, 22.0])
    val = safe_wmape(y_true, y_pred)
    # sum(|err|) = 2 + 2 + 2 = 6, sum(y_true) = 30 -> 6/30 = 0.20
    assert pytest.approx(val, 0.01) == 0.20

    # All zeros
    assert safe_wmape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0


def test_predict_demand_non_negative() -> None:
    """Verify DemandModel output is strictly bounded >= 0."""
    # Dummy linear model with negative prediction tendency
    lr = LinearRegression()
    dummy_X = pd.DataFrame(np.ones((5, len(DEFAULT_DEMAND_FEATURES))), columns=DEFAULT_DEMAND_FEATURES)
    dummy_y = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    lr.fit(dummy_X, dummy_y)
    lr.coef_ = np.array([-100.0] + [0.0] * (len(DEFAULT_DEMAND_FEATURES) - 1))
    lr.intercept_ = 10.0

    model = DemandModel(
        model=lr,
        feature_names=DEFAULT_DEMAND_FEATURES,
        min_observed_price=1.0,
        max_observed_price=2.0,
    )

    ctx = {f: 1.0 for f in DEFAULT_DEMAND_FEATURES}
    # Large price causes unconstrained linear model to predict negative
    pred = model.predict_demand(price=10.0, context_features=ctx)
    assert pred >= 0.0
    assert pred == 0.0


def test_price_response_curve_extrapolation_warning() -> None:
    """Verify price_response_curve warns when price exceeds observed historical range."""
    lr = LinearRegression()
    dummy_X = pd.DataFrame(np.ones((5, len(DEFAULT_DEMAND_FEATURES))), columns=DEFAULT_DEMAND_FEATURES)
    dummy_y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    lr.fit(dummy_X, dummy_y)
    lr.coef_ = np.ones(len(DEFAULT_DEMAND_FEATURES))
    lr.intercept_ = 5.0

    model = DemandModel(
        model=lr,
        feature_names=DEFAULT_DEMAND_FEATURES,
        min_observed_price=1.00,
        max_observed_price=2.00,
    )

    ctx = {f: 1.0 for f in DEFAULT_DEMAND_FEATURES}

    # Should trigger extrapolation warning when evaluating outside [1.00, 2.00]
    with pytest.warns(UserWarning, match="outside historical observed range"):
        curve = model.price_response_curve(ctx, min_price=0.50, max_price=3.00, n_points=5)

    assert len(curve) == 5
    assert "price" in curve.columns
    assert "predicted_demand" in curve.columns
    assert "is_extrapolated" in curve.columns
    assert curve["is_extrapolated"].any()


def test_log_log_elasticity_estimation(demand_test_df: pd.DataFrame) -> None:
    """Verify log-log regression computes elasticity, standard error, and CI."""
    sub = demand_test_df[demand_test_df["item_id"] == "FOODS_3_090"]
    res = estimate_log_log_elasticity(sub, item_id="FOODS_3_090", store_id="CA_1")

    assert "elasticity" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["ci_upper"]
    assert "caveat" in res
    assert "observational" in res["caveat"].lower()


def test_linear_approximation_parameters(demand_test_df: pd.DataFrame) -> None:
    """Verify linear approximation yields structural parameters a and b."""
    res = fit_linear_approximation(demand_test_df, item_id="FOODS_3_090", store_id="CA_1")

    assert res["item_id"] == "FOODS_3_090"
    assert res["store_id"] == "CA_1"
    assert "a" in res
    assert "b" in res
    assert "c" in res
    assert res["b"] > 0.0  # Downward sloping demand enforced
    assert res["a"] > 0.0


def test_model_serialization(tmp_path: Path) -> None:
    """Verify saving and loading DemandModel wrapper retains features and bounds."""
    lr = LinearRegression()
    lr.coef_ = np.ones(len(DEFAULT_DEMAND_FEATURES))
    lr.intercept_ = 12.5

    wrapper = DemandModel(
        model=lr,
        feature_names=DEFAULT_DEMAND_FEATURES,
        min_observed_price=1.20,
        max_observed_price=3.50,
        model_name="TestLR",
    )

    save_path = tmp_path / "test_demand.pkl"
    save_demand_model(wrapper, path=save_path)
    assert save_path.exists()

    loaded = load_demand_model(path=save_path)
    assert loaded.model_name == "TestLR"
    assert loaded.min_observed_price == 1.20
    assert loaded.max_observed_price == 3.50
    assert loaded.feature_names == DEFAULT_DEMAND_FEATURES
