"""Market Intelligence Dashboard Page for PricePilot."""

import streamlit as st
import pandas as pd

from dashboard.utils import (
    load_processed_dataset,
    load_market_kpis_table,
    safe_scalar,
)
from src.data.eda import (
    compute_market_kpis,
    plot_sales_and_price_over_time,
    plot_price_vs_quantity,
    plot_seasonality,
    plot_promotion_effect,
    plot_price_change_vs_sales_change,
    plot_product_comparison,
)


def render() -> None:
    """Render the Market Intelligence page."""
    st.title("Market Intelligence & Exploratory Analysis")
    st.markdown(
        """
        Explore empirical price and demand dynamics across selected Walmart M5 items in category **FOODS_3**
        across store **CA_1** (Leader Store) and store **CA_2** (Follower/Competitor Store).
        """
    )

    df = load_processed_dataset()
    if df.empty:
        st.error("Processed data could not be loaded. Please ensure data/processed/m5_processed.parquet exists.")
        return

    # Sidebar / Page Filters
    st.subheader("Market Scope Selector")
    col1, col2, col3 = st.columns(3)

    available_items = sorted(df["item_id"].unique().tolist())
    available_stores = sorted(df["store_id"].unique().tolist())

    with col1:
        selected_item = st.selectbox(
            "Select Item",
            options=available_items,
            index=available_items.index("FOODS_3_090") if "FOODS_3_090" in available_items else 0,
        )
    with col2:
        selected_store = st.selectbox(
            "Select Store",
            options=available_stores,
            index=available_stores.index("CA_1") if "CA_1" in available_stores else 0,
        )
    with col3:
        log_scale = st.checkbox("Log-log Empirical Demand Curve", value=True)

    # Compute & Display KPIs for selected item & store
    kpi_df = compute_market_kpis(df, item_id=selected_item, store_id=selected_store)
    kpi_dict = kpi_df.iloc[0].to_dict() if not kpi_df.empty else {}

    avg_price = float(safe_scalar(kpi_dict.get("avg_price", 0.0), 0.0))
    avg_demand = float(safe_scalar(kpi_dict.get("avg_daily_demand", 0.0), 0.0))
    price_cv = float(safe_scalar(kpi_dict.get("price_cv", 0.0), 0.0))
    price_regime = str(kpi_dict.get("price_volatility", kpi_dict.get("volatility_category", "N/A")))
    demand_slope = float(safe_scalar(kpi_dict.get("demand_slope", kpi_dict.get("demand_trend_slope", 0.0)), 0.0))
    trend_label = str(kpi_dict.get("demand_trend", "Stable"))

    st.markdown("### Market Intelligence KPIs")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Average Selling Price", f"${avg_price:.2f}")
    with m2:
        st.metric("Average Daily Demand", f"{avg_demand:.1f} units")
    with m3:
        st.metric("Price Volatility (CV)", f"{price_cv:.4f}")
    with m4:
        st.metric("Price Regime", price_regime)
    with m5:
        st.metric("Demand Trend", f"{trend_label} ({demand_slope:+.3f}/day)")

    st.markdown("---")

    # Time series of sales and price
    st.subheader("Sales & Price Trends Over Time")
    fig_time = plot_sales_and_price_over_time(df, item_id=selected_item, store_id=selected_store)
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # Empirical Demand & Seasonality
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Empirical Price-Quantity Relationship")
        fig_scatter = plot_price_vs_quantity(df, item_id=selected_item, store_id=selected_store, log_scale=log_scale)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        st.subheader("Seasonality (Day-of-Week & Month)")
        fig_season = plot_seasonality(df, item_id=selected_item, store_id=selected_store)
        st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("---")

    # Promotion Effects & Price Change Responses
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Promotion / Price-Discount Effect")
        st.caption("Sales when price is below item historical mean vs average/above")
        fig_promo = plot_promotion_effect(df, item_id=selected_item, store_id=selected_store)
        st.plotly_chart(fig_promo, use_container_width=True)
    with p2:
        st.subheader("Price Change vs Sales Change")
        st.caption("Period-over-period percentage price shock vs percentage volume shock")
        fig_change = plot_price_change_vs_sales_change(df, item_id=selected_item, store_id=selected_store)
        st.plotly_chart(fig_change, use_container_width=True)

    st.markdown("---")

    # Product Comparison Across Catalog
    st.subheader("Cross-Product Catalog Comparison")
    st.caption("Normalized comparison of top items in store")
    compare_items = st.multiselect(
        "Select products to compare",
        options=available_items,
        default=[it for it in ["FOODS_3_090", "FOODS_3_723", "FOODS_3_491", "FOODS_3_124"] if it in available_items][:4],
    )
    if compare_items:
        df_store = df[df["store_id"] == selected_store] if "store_id" in df.columns else df
        fig_comp = plot_product_comparison(df_store, item_ids=compare_items)
        st.plotly_chart(fig_comp, use_container_width=True)
