"""Smoke test validating dashboard data loading logic and page rendering against real outputs."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.utils import (
    get_cached_competitor_model,
    get_cached_demand_model,
    load_elasticity_table,
    load_linear_demand_parameters,
    load_market_kpis_table,
    load_model_comparison_table,
    load_processed_dataset,
    load_sensitivity_datasets,
    load_strategy_comparison_data,
    safe_scalar,
    safe_style_dataframe,
)
from src.data.eda import compute_market_kpis


@pytest.fixture(autouse=True)
def clear_caches_for_smoke_tests():
    """Ensure clean caches for smoke tests."""
    load_processed_dataset.clear()
    load_strategy_comparison_data.clear()
    load_sensitivity_datasets.clear()
    load_model_comparison_table.clear()
    load_elasticity_table.clear()
    load_market_kpis_table.clear()
    yield
    load_processed_dataset.clear()
    load_strategy_comparison_data.clear()
    load_sensitivity_datasets.clear()


def test_safe_scalar_coercion() -> None:
    """Verify safe_scalar coerces diverse container types to scalar values."""
    assert safe_scalar(5.5) == 5.5
    assert safe_scalar("test") == "test"
    assert safe_scalar(pd.Series([1.23, 4.56])) == 1.23
    assert safe_scalar(pd.Series([]), default=0.0) == 0.0
    assert safe_scalar(pd.DataFrame({"a": [9.99]})) == 9.99
    assert safe_scalar(pd.DataFrame(), default=0.0) == 0.0
    assert safe_scalar([42, 99]) == 42
    assert safe_scalar([], default=0.0) == 0.0
    assert safe_scalar(np.array([7.7, 8.8])) == 7.7
    assert safe_scalar(None, default=0.0) == 0.0
    assert safe_scalar(np.nan, default=0.0) == 0.0


def test_safe_style_dataframe() -> None:
    """Verify safe_style_dataframe handles missing columns gracefully without KeyError."""
    df = pd.DataFrame({
        "model": ["M1", "M2"],
        "test_mae": [1.0, 2.0],
        "test_rmse": [2.0, 3.0],
    })
    # Subset contains columns that exist ('test_mae') and non-existent ('Test RMSE', 'missing_col')
    styler = safe_style_dataframe(
        df,
        min_cols=["test_mae", "missing_col", "Test RMSE"],
        max_cols=["missing_max"],
    )
    assert styler is not None


def test_processed_dataset_loading() -> None:
    """Verify load_processed_dataset retrieves valid M5 dataframe."""
    df = load_processed_dataset()
    assert not df.empty
    assert "date" in df.columns
    assert "item_id" in df.columns
    assert "store_id" in df.columns
    assert "sell_price" in df.columns
    assert "sales" in df.columns
    assert len(df) > 1000


def test_market_kpi_computation_and_scalar_metrics() -> None:
    """Verify compute_market_kpis output format and extraction of scalar metrics."""
    df = load_processed_dataset()
    kpi_df = compute_market_kpis(df, item_id="FOODS_3_090", store_id="CA_1")

    assert isinstance(kpi_df, pd.DataFrame)
    assert len(kpi_df) == 1

    kpi_dict = kpi_df.iloc[0].to_dict()
    assert "avg_price" in kpi_dict
    assert "avg_daily_demand" in kpi_dict
    assert "price_cv" in kpi_dict
    assert "price_volatility" in kpi_dict
    assert "demand_slope" in kpi_dict
    assert "demand_trend" in kpi_dict

    # Check scalar formatting succeeds without Series format error
    avg_price = float(safe_scalar(kpi_dict.get("avg_price", 0.0), 0.0))
    avg_demand = float(safe_scalar(kpi_dict.get("avg_daily_demand", 0.0), 0.0))
    price_cv = float(safe_scalar(kpi_dict.get("price_cv", 0.0), 0.0))

    assert f"${avg_price:.2f}" == "$1.38"
    assert f"{avg_demand:.1f}" == "66.5"
    assert f"{price_cv:.4f}" == "0.0915"


def test_model_comparison_loading_and_styling() -> None:
    """Verify load_model_comparison_table columns and safe styling."""
    df_models = load_model_comparison_table()
    assert not df_models.empty
    assert "model" in df_models.columns
    assert "test_mae" in df_models.columns
    assert "test_rmse" in df_models.columns
    assert "test_r2" in df_models.columns

    styler = safe_style_dataframe(
        df_models,
        min_cols=["test_mae", "test_rmse", "Test MAE", "Test RMSE"],
        max_cols=["test_r2", "Test R2"],
    )
    assert styler is not None


def test_elasticity_table_loading() -> None:
    """Verify load_elasticity_table returns expected econometric columns."""
    df_elast = load_elasticity_table()
    assert not df_elast.empty
    assert "item_id" in df_elast.columns
    assert "store_id" in df_elast.columns
    assert "elasticity" in df_elast.columns
    assert "ci_lower" in df_elast.columns
    assert "ci_upper" in df_elast.columns
    assert "p_value" in df_elast.columns
    # Check that either r2 or r_squared is present
    assert any(c in df_elast.columns for c in ["r2", "r_squared"])


def test_strategy_comparison_data_loading() -> None:
    """Verify precomputed simulation datasets have matching schema."""
    df_summary, df_traj, df_runs = load_strategy_comparison_data()
    assert not df_summary.empty
    assert not df_traj.empty
    assert not df_runs.empty

    for req in ["strategy", "mean_total_profit", "ci_95_lower", "ci_95_upper", "profit_improvement_pct"]:
        assert req in df_summary.columns


def test_sensitivity_datasets_loading() -> None:
    """Verify all sensitivity analysis sweep files load properly."""
    dfs = load_sensitivity_datasets()
    for key in ["heatmap", "accuracy_curve", "behaviors", "demand", "robustness"]:
        assert key in dfs
        assert not dfs[key].empty


def test_demand_model_and_price_response_curve() -> None:
    """Verify cached demand model generates response curve with context features."""
    model_wrapper = get_cached_demand_model()
    assert model_wrapper is not None
    assert hasattr(model_wrapper, "price_response_curve")

    context = {"lag_sales_1": 66.0, "rolling_mean_7": 65.0, "is_weekend": 0}
    grid = np.array([1.20, 1.38, 1.50])
    curve_df = model_wrapper.price_response_curve(context_features=context, price_grid=grid)

    assert not curve_df.empty
    assert "price" in curve_df.columns
    assert "predicted_demand" in curve_df.columns
    assert len(curve_df) == len(grid)


def test_competitor_model_and_metrics_attribute() -> None:
    """Verify cached competitor response model handles metrics attribute safely."""
    comp_model = get_cached_competitor_model()
    assert comp_model is not None

    # Safe extraction should never raise AttributeError
    metrics = getattr(comp_model, "metrics", {}) or {}
    clf_acc = float(safe_scalar(metrics.get("clf_accuracy", 0.999), 0.999))
    assert 0.0 <= clf_acc <= 1.0


def test_app_clickthrough_smoke() -> None:
    """End-to-end smoke test clicking through every page in the Streamlit app."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)
    at.run()
    assert not at.exception

    nav = at.sidebar.radio[0]
    for page_name in nav.options:
        nav.set_value(page_name)
        at.run()
        assert not at.exception, f"Exception occurred while rendering page: {page_name}"
