"""Demand Modeling & Elasticity Dashboard Page for PricePilot."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils import (
    load_processed_dataset,
    get_cached_demand_model,
    load_model_comparison_table,
    load_elasticity_table,
    load_linear_demand_parameters,
)
from src.demand.elasticity import plot_elasticity_comparison, ELASTICITY_CAVEAT
from src.demand.forecasting import plot_actual_vs_predicted


def render() -> None:
    """Render the Demand Forecasting page."""
    st.title("Demand Modeling & Elasticity Estimation")
    st.markdown(
        """
        Evaluate customer price-sensitivity using machine learning regression and econometric
        log-log models on historical Walmart M5 sales data.
        """
    )

    # 1. Model Comparison Table
    st.subheader("1. Forecasting Model Comparison")
    st.caption("Chronological Out-of-Sample Performance on Validation (28 days) and Test (28 days) Sets")

    df_models = load_model_comparison_table()
    if not df_models.empty:
        st.dataframe(
            df_models.style.highlight_min(subset=["Test MAE", "Test RMSE"], color="#d4edda")
            .highlight_max(subset=["Test R2"], color="#d4edda"),
            use_container_width=True,
        )
        st.success(
            "**Selected Champion Model**: **Hist Gradient Boosting Regressor** "
            "(Test MAE: 3.48, Test RMSE: 5.92, Test R²: 0.8081). Serialized to `models/demand_model.pkl`."
        )
    else:
        st.info("Model comparison data not found.")

    st.markdown("---")

    # 2. Actual vs Predicted Demand
    st.subheader("2. Actual vs Predicted Demand (Test Horizon)")
    df = load_processed_dataset()
    model_wrapper = get_cached_demand_model()

    if not df.empty and model_wrapper is not None:
        sub = df[(df["item_id"] == "FOODS_3_090") & (df["store_id"] == "CA_1")].sort_values("date")
        test_sub = sub.tail(28).copy()

        if len(test_sub) > 0 and hasattr(model_wrapper, "model") and hasattr(model_wrapper, "feature_names"):
            features = [c for c in model_wrapper.feature_names if c in test_sub.columns]
            preds = model_wrapper.model.predict(test_sub[features])
            fig_pred = plot_actual_vs_predicted(
                actual=test_sub["sales"],
                predicted=preds,
                dates=test_sub["date"],
                title="FOODS_3_090 (CA_1) - Out-of-Sample Test Set (Last 28 Days)",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Demand model or dataset unavailable.")

    st.markdown("---")

    # 3. Elasticity Estimates with Observational Caveat
    st.subheader("3. Price Elasticity Estimates (Log-Log Econometric Regression)")

    st.warning(
        f"**IMPORTANT METHODOLOGICAL CAVEAT (CONTEXT.md Rule #4)**:\n\n{ELASTICITY_CAVEAT}"
    )

    df_elast = load_elasticity_table()
    if not df_elast.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(
                df_elast[["item_id", "store_id", "elasticity", "ci_lower", "ci_upper", "p_value", "r_squared"]],
                use_container_width=True,
                height=350,
            )
        with col2:
            fig_el = plot_elasticity_comparison(df_elast)
            st.plotly_chart(fig_el, use_container_width=True)

    st.markdown("---")

    # 4. Interactive Demand vs Price Response Curve
    st.subheader("4. Price-Response Curve & Extrapolation Bounds")
    st.caption("Inspect anticipated demand response across candidate price points.")

    if model_wrapper is not None:
        p_min_obs = model_wrapper.min_observed_price
        p_max_obs = model_wrapper.max_observed_price

        c1, c2, c3 = st.columns(3)
        with c1:
            test_min_p = st.number_input("Grid Min Price ($)", min_value=0.50, max_value=2.50, value=max(0.50, p_min_obs * 0.75), step=0.05)
        with c2:
            test_max_p = st.number_input("Grid Max Price ($)", min_value=0.50, max_value=3.00, value=min(3.00, p_max_obs * 1.25), step=0.05)
        with c3:
            st.metric("Observed Price Range", f"${p_min_obs:.2f} – ${p_max_obs:.2f}")

        grid_prices = np.linspace(test_min_p, test_max_p, 40)
        curve_df = model_wrapper.price_response_curve(price_grid=grid_prices)

        fig_curve = px.line(
            curve_df,
            x="price",
            y="predicted_demand",
            title=f"Price-Response Curve for {model_wrapper.item_id} ({model_wrapper.store_id})",
            labels={"price": "Unit Price ($)", "predicted_demand": "Predicted Daily Demand (units)"},
            template="plotly_white",
        )

        # Highlight extrapolation bounds
        fig_curve.add_vrect(
            x0=p_min_obs,
            x1=p_max_obs,
            fillcolor="green",
            opacity=0.08,
            line_width=1,
            annotation_text="Empirically Observed Price Region",
            annotation_position="top left",
        )

        st.plotly_chart(fig_curve, use_container_width=True)

        if (test_min_p < p_min_obs) or (test_max_p > p_max_obs):
            st.warning(
                f"**Extrapolation Warning**: Shaded region represents the historical observed price range "
                f"([${p_min_obs:.2f}, ${p_max_obs:.2f}]). Decisions outside this region rely on model extrapolation."
            )
