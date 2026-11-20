"""Tests for Phase 2 Exploratory Data Analysis module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data.eda import (
    compute_market_kpis,
    plot_price_change_vs_sales_change,
    plot_price_vs_quantity,
    plot_product_comparison,
    plot_promotion_effect,
    plot_sales_and_price_over_time,
    plot_sales_distribution,
    plot_seasonality,
)
from src.utils.config import DATA_SAMPLE_DIR


@pytest.fixture(scope="module")
def sample_eda_df() -> pd.DataFrame:
    """Create a synthetic time series DataFrame matching the processed schema."""
    dates = pd.date_range("2015-01-01", periods=60, freq="D")
    records = []

    for item in ["ITEM_1", "ITEM_2"]:
        for store in ["STORE_1", "STORE_2"]:
            base_p = 2.0 if item == "ITEM_1" else 5.0
            for i, d in enumerate(dates):
                # Price changes periodically
                price = base_p + (0.5 if (i // 15) % 2 == 1 else 0.0)
                sales = max(0.0, 20.0 - 3.0 * price + np.random.normal(0, 1))
                records.append({
                    "item_id": item,
                    "store_id": store,
                    "date": d,
                    "wm_yr_wk": 11501 + (i // 7),
                    "sales": float(sales),
                    "sell_price": float(price),
                    "weekday": d.day_name(),
                    "month": d.month,
                    "is_discounted": 1 if price < (base_p + 0.25) else 0,
                    "has_event": 1 if i % 10 == 0 else 0,
                    "log_price": float(np.log(price)),
                    "log_sales": float(np.log1p(sales)),
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_compute_market_kpis(sample_eda_df: pd.DataFrame) -> None:
    """Verify market KPIs are calculated with expected fields and types."""
    kpis = compute_market_kpis(sample_eda_df)
    assert not kpis.empty
    expected_cols = [
        "item_id", "store_id", "avg_price", "price_cv", "price_volatility",
        "distinct_prices", "avg_daily_demand", "zero_sales_share",
        "demand_trend", "recommended_for_modeling"
    ]
    for col in expected_cols:
        assert col in kpis.columns

    # Volatility labels should be strictly valid
    assert set(kpis["price_volatility"].unique()).issubset({"Low", "Moderate", "High"})
    assert set(kpis["demand_trend"].unique()).issubset({"Increasing", "Decreasing", "Stable"})


def test_plot_sales_and_price_over_time(sample_eda_df: pd.DataFrame) -> None:
    """Verify dual-axis time-series figure structure."""
    fig = plot_sales_and_price_over_time(sample_eda_df, "ITEM_1", "STORE_1")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # One bar, one scatter line


def test_plot_price_vs_quantity(sample_eda_df: pd.DataFrame) -> None:
    """Verify price vs quantity scatter plot in both natural and log scales."""
    fig_nat = plot_price_vs_quantity(sample_eda_df, "ITEM_1", "STORE_1", log_scale=False)
    assert isinstance(fig_nat, go.Figure)

    fig_log = plot_price_vs_quantity(sample_eda_df, "ITEM_1", "STORE_1", log_scale=True)
    assert isinstance(fig_log, go.Figure)


def test_plot_sales_distribution(sample_eda_df: pd.DataFrame) -> None:
    """Verify distribution figure."""
    fig = plot_sales_distribution(sample_eda_df, "ITEM_1", "STORE_1")
    assert isinstance(fig, go.Figure)


def test_plot_seasonality(sample_eda_df: pd.DataFrame) -> None:
    """Verify seasonality plots across dow, month, event."""
    for freq in ["dow", "month", "event"]:
        fig = plot_seasonality(sample_eda_df, "ITEM_1", "STORE_1", freq=freq)
        assert isinstance(fig, go.Figure)


def test_plot_price_change_vs_sales_change(sample_eda_df: pd.DataFrame) -> None:
    """Verify price change vs sales change scatter."""
    fig = plot_price_change_vs_sales_change(sample_eda_df, "ITEM_1", "STORE_1")
    assert isinstance(fig, go.Figure)


def test_plot_promotion_effect(sample_eda_df: pd.DataFrame) -> None:
    """Verify promotion effect figure."""
    fig = plot_promotion_effect(sample_eda_df, "ITEM_1", "STORE_1")
    assert isinstance(fig, go.Figure)


def test_plot_product_comparison(sample_eda_df: pd.DataFrame) -> None:
    """Verify product comparison figure."""
    fig_sales = plot_product_comparison(sample_eda_df, metric="sales")
    assert isinstance(fig_sales, go.Figure)
    fig_price = plot_product_comparison(sample_eda_df, metric="price")
    assert isinstance(fig_price, go.Figure)
