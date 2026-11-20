"""Sensitivity Analysis Dashboard Page for PricePilot."""

import streamlit as st
import pandas as pd

from dashboard.utils import load_sensitivity_datasets
from src.experiments.sensitivity import (
    plot_heatmap_strength_x_accuracy,
    plot_accuracy_vs_profit_curve,
    plot_competitor_behavior_comparison,
    plot_demand_sensitivity_vs_price,
    plot_robustness_analysis,
)


def render() -> None:
    """Render the Sensitivity Analysis page."""
    st.title("Sensitivity & Robustness Analysis")
    st.markdown(
        """
        Explore precomputed multidimensional sensitivity sweeps to assess under what conditions
        ML-aware Stackelberg pricing succeeds, degrades, or underperforms baseline pricing.
        """
    )

    data = load_sensitivity_datasets()

    df_heatmap = data.get("heatmap", pd.DataFrame())
    df_acc = data.get("accuracy_curve", pd.DataFrame())
    df_beh = data.get("behaviors", pd.DataFrame())
    df_dem = data.get("demand", pd.DataFrame())
    df_rob = data.get("robustness", pd.DataFrame())

    if df_heatmap.empty:
        st.error("Sensitivity data not found. Please run notebooks/run_sensitivity_analysis.py.")
        return

    # 1. Heatmap: Response Strength x Accuracy
    st.subheader("1. Profit Improvement Heatmap: Competitor Response Strength vs ML Accuracy")
    st.caption("Gain % (Strategy C vs A) across competitor reaction strength and ML model accuracy.")
    fig_heat = plot_heatmap_strength_x_accuracy(df_heatmap)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # 2. Accuracy vs Profit Curve with Break-Even Threshold
    st.subheader("2. ML Accuracy vs Profit: Symmetric vs Asymmetric Error Regimes")
    st.caption("Demonstrating why error direction (optimistic vs pessimistic) matters more than accuracy alone.")
    if not df_acc.empty:
        fig_curve = plot_accuracy_vs_profit_curve(df_acc)
        st.plotly_chart(fig_curve, use_container_width=True)

        st.info(
            """
            **Break-Even Accuracy Analysis**:
            - **Symmetric Errors**: Break-even occurs at **~58%–62% accuracy**. Above 65%, ML-aware pricing consistently avoids error.
            - **Optimistic Errors** (overestimating follower matching): Threshold rises to **≥ 75%** because unreciprocated price hikes lead to severe volume losses.
            - **Pessimistic Errors** (underestimating follower matching): Threshold stays low (**~50%**) because the optimizer cautiously stays near baseline, risking only foregone upside.
            """
        )

    st.markdown("---")

    # 3. Competitor Behaviors & Response Latency Delays
    st.subheader("3. Realized Profit Across Competitor Policies & Reaction Latencies")
    if not df_beh.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_beh = plot_competitor_behavior_comparison(df_beh)
            st.plotly_chart(fig_beh, use_container_width=True)
        with c2:
            st.dataframe(df_beh, use_container_width=True, height=400)

    st.markdown("---")

    # 4. Demand Elasticity Sensitivity & Robustness
    st.subheader("4. Structural Demand Elasticity & Robustness Under Uncertainty")
    d1, d2 = st.columns(2)
    with d1:
        if not df_dem.empty:
            fig_dem = plot_demand_sensitivity_vs_price(df_dem)
            st.plotly_chart(fig_dem, use_container_width=True)
    with d2:
        if not df_rob.empty:
            fig_rob = plot_robustness_analysis(df_rob)
            st.plotly_chart(fig_rob, use_container_width=True)

    st.markdown("---")

    # Core Strategic Takeaways
    st.subheader("5. Key Research Answers")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            r"""
            #### When does ML-aware pricing help?
            - When the competitor reacts responsively ($\ge 0.75$) and follows price leadership (`match` or `raise`).
            - When cross-price elasticity is high ($c \ge 0.50$), allowing margin expansion without share erosion.
            """
        )
        st.markdown(
            r"""
            #### What is the accuracy threshold?
            - Symmetrically: ~60% accuracy to beat baseline.
            - Highly asymmetric: requires $\ge 75\%$ if errors are optimistic, but only ~50% if pessimistic.
            """
        )
    with q2:
        st.markdown(
            r"""
            #### When does it hurt?
            - In price-war regimes (`undercut`), where leader price increases surrender volume.
            - When competitor reaction is delayed ($\ge 3$ periods), leaving the leader exposed.
            - In asymmetric optimistic error regimes.
            """
        )
        st.markdown(
            r"""
            #### How robust is the recommendation?
            - Very robust across unit cost levels ($C \in [\$0.65, \$1.05]$) and moderate demand volatility ($\sigma_d \le 2.0$).
            - Bounded price grids ($[0.70 P_{base}, 1.30 P_{base}]$) safeguard against model divergence.
            """
        )
