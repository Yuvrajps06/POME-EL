"""Price Optimization Engine Dashboard Page for PricePilot."""

import streamlit as st
import numpy as np
import pandas as pd

from dashboard.utils import get_cached_competitor_model
from src.optimization.baseline_optimizer import optimize_baseline, plot_profit_vs_price
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg


def render() -> None:
    """Render the Price Optimizer page."""
    st.title("Stackelberg Game Theoretic Price Optimizer")
    st.markdown(
        """
        Compute optimal pricing decisions comparing:
        - **Strategy A (Traditional Baseline)**: Ignores competitor reaction.
        - **Strategy B (Game-Theoretic Stackelberg)**: Anticipates rule-based competitor response.
        - **Strategy C (ML-Aware Stackelberg)**: Anticipates machine-learning predicted competitor response.
        """
    )

    st.markdown("---")

    # Input controls
    st.subheader("1. Pricing Decision Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        current_price = st.number_input("Current Firm Price ($)", min_value=0.50, max_value=5.00, value=1.38, step=0.01)
    with c2:
        unit_cost = st.number_input(
            "Exogenous Unit Cost ($)*",
            min_value=0.10,
            max_value=4.50,
            value=0.83,
            step=0.01,
            help="Cost is strictly an exogenous input parameter and never inferred from selling price.",
        )
    with c3:
        current_follower_price = st.number_input("Competitor Price ($)", min_value=0.50, max_value=5.00, value=1.37, step=0.01)
    with c4:
        grid_range_pct = st.slider("Price Search Range (± %)", min_value=10, max_value=50, value=30, step=5)

    st.caption("*Methodological requirement (CONTEXT.md Rule #2): Walmart M5 data contains no unit costs; cost is strictly an external input parameter.")

    # Advanced parameters expander
    with st.expander("Demand Function & Behavioral Parameters (Advanced)", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            a_param = st.number_input("Demand Intercept (a)", min_value=10.0, max_value=500.0, value=165.39, step=1.0)
        with a2:
            b_param = st.number_input("Own-Price Sensitivity (b)", min_value=1.0, max_value=300.0, value=79.18, step=1.0)
        with a3:
            c_param = st.number_input("Cross-Price Sensitivity (c)", min_value=0.0, max_value=5.0, value=0.50, step=0.05)
        with a4:
            rule_beh = st.selectbox("Strategy B Follower Policy", options=["match", "undercut", "maintain", "raise", "adaptive"], index=0)

    # Optimization Execution
    min_factor = 1.0 - (grid_range_pct / 100.0)
    max_factor = 1.0 + (grid_range_pct / 100.0)

    comp_model = get_cached_competitor_model()

    res_a = optimize_baseline(
        baseline_price=current_price,
        current_follower_price=current_follower_price,
        cost=unit_cost,
        a=a_param,
        b=b_param,
        c=c_param,
        min_factor=min_factor,
        max_factor=max_factor,
        num_steps=61,
    )
    res_b = optimize_game_rule(
        baseline_price=current_price,
        current_follower_price=current_follower_price,
        cost=unit_cost,
        a=a_param,
        b=b_param,
        c=c_param,
        rule_params={"behavior": rule_beh},
        min_factor=min_factor,
        max_factor=max_factor,
        num_steps=61,
    )
    res_c = optimize_ml_stackelberg(
        baseline_price=current_price,
        current_follower_price=current_follower_price,
        cost=unit_cost,
        a=a_param,
        b=b_param,
        c=c_param,
        competitor_model=comp_model,
        min_factor=min_factor,
        max_factor=max_factor,
        num_steps=61,
    )

    st.markdown("---")

    # Comparative Results Display
    st.subheader("2. Optimization Results by Strategy")
    colA, colB, colC = st.columns(3)

    with colA:
        st.info("### Strategy A (Baseline)")
        st.metric("Recommended Price", f"${res_a.recommended_price:.2f}")
        st.metric("Expected Daily Profit", f"${res_a.expected_profit:.2f}")
        st.metric("Expected Daily Demand", f"{res_a.expected_demand:.1f} units")
        st.metric("Predicted Follower Price", f"${res_a.predicted_follower_price:.2f} (Static)")
        st.metric("Unit Margin", f"${res_a.unit_margin:.2f} ({res_a.margin_pct*100:.1f}%)")

    with colB:
        st.warning(f"### Strategy B (Rule: {rule_beh.capitalize()})")
        st.metric("Recommended Price", f"${res_b.recommended_price:.2f}")
        st.metric("Expected Daily Profit", f"${res_b.expected_profit:.2f}")
        st.metric("Expected Daily Demand", f"{res_b.expected_demand:.1f} units")
        st.metric("Predicted Follower Price", f"${res_b.predicted_follower_price:.2f}")
        st.metric("Unit Margin", f"${res_b.unit_margin:.2f} ({res_b.margin_pct*100:.1f}%)")

    with colC:
        st.success("### Strategy C (ML-Aware Stackelberg)")
        st.metric("Recommended Price", f"${res_c.recommended_price:.2f}")
        st.metric("Expected Daily Profit", f"${res_c.expected_profit:.2f}")
        st.metric("Expected Daily Demand", f"{res_c.expected_demand:.1f} units")
        st.metric("Predicted Follower Price", f"${res_c.predicted_follower_price:.2f} (ML)")
        st.metric("Unit Margin", f"${res_c.unit_margin:.2f} ({res_c.margin_pct*100:.1f}%)")

    # Boundary or edge case alerts
    for r in [res_a, res_b, res_c]:
        if r.boundary_warning:
            st.warning(f"⚠️ **{r.strategy}**: Optimum reached candidate grid boundary (${r.recommended_price:.2f}). Consider expanding search range.")
        if r.negative_margin:
            st.error(f"🚨 **{r.strategy}**: Recommendation has negative margin (Price ${r.recommended_price:.2f} < Cost ${r.cost:.2f}).")

    st.markdown("---")

    # Profit vs Price Graph
    st.subheader("3. Profit vs Candidate Price Curves")
    st.caption("Visual inspection of profit curves across candidate price search grid.")

    fig_profit = plot_profit_vs_price([res_a, res_b, res_c])
    st.plotly_chart(fig_profit, use_container_width=True)

    # Detailed Grid Table expander
    with st.expander("View Full Evaluated Price Grid Data"):
        st.write("Strategy C Grid Evaluations:")
        st.dataframe(res_c.grid_evaluations, use_container_width=True)
