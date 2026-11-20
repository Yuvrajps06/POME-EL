"""Executive Dashboard Page for PricePilot."""

import streamlit as st
import pandas as pd

from dashboard.utils import (
    load_linear_demand_parameters,
    load_strategy_comparison_data,
    get_cached_competitor_model,
)
from src.experiments.comparison import plot_strategy_comparison_bars, plot_cumulative_profit
from src.optimization.baseline_optimizer import optimize_baseline
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg


def render() -> None:
    """Render the Executive Summary page."""
    st.title("PricePilot Executive Summary")
    st.markdown(
        """
        **Strategic Pricing Decision Support System** integrating Walmart M5 empirical demand modeling,
        machine learning competitor response prediction, and Stackelberg game theory.
        """
    )

    st.markdown("---")

    # Load precomputed benchmarks and hero item defaults
    df_summary, df_traj, df_runs = load_strategy_comparison_data()
    comp_model = get_cached_competitor_model()

    # Hero Item Context: FOODS_3_090 at CA_1 (Cost = $0.83, Base = $1.38, Competitor = $1.37)
    current_price = 1.38
    competitor_price = 1.37
    cost = 0.83
    a = 165.39
    b = 79.18
    c = 0.50

    # Real-time optimization outputs for hero product
    res_a = optimize_baseline(current_price, competitor_price, cost, a, b, c)
    res_b = optimize_game_rule(current_price, competitor_price, cost, a, b, c, rule_params={"behavior": "match"})
    res_c = optimize_ml_stackelberg(current_price, competitor_price, cost, a, b, c, competitor_model=comp_model)

    profit_improvement_pct = ((res_c.expected_profit - res_a.expected_profit) / res_a.expected_profit) * 100.0

    st.subheader("Key Strategic Pricing Metrics (Hero Item: FOODS_3_090, CA_1)")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta="Walmart CA_1",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Recommended Price (Strategy C)",
            value=f"${res_c.recommended_price:.2f}",
            delta=f"+${res_c.recommended_price - current_price:.2f} (+{((res_c.recommended_price - current_price)/current_price)*100:.1f}%)",
        )
    with col3:
        st.metric(
            label="Predicted Competitor Price",
            value=f"${res_c.predicted_follower_price:.2f}",
            delta=f"+${res_c.predicted_follower_price - competitor_price:.2f} (Follower)",
        )
    with col4:
        st.metric(
            label="Expected Daily Profit",
            value=f"${res_c.expected_profit:.2f}",
            delta=f"+{profit_improvement_pct:.2f}% vs Baseline",
        )
    with col5:
        st.metric(
            label="Expected Daily Demand",
            value=f"{res_c.expected_demand:.1f} units",
            delta="Demand Trend: Stable (-0.007/day)",
            delta_color="off",
        )

    st.markdown("---")

    # Strategy Comparison Tiles
    st.subheader("Leader Strategy Benchmark Comparison")
    scol1, scol2, scol3 = st.columns(3)

    with scol1:
        st.info("### Strategy A: Traditional Baseline")
        st.write("Maximizes profit ignoring competitor price reactions.")
        st.write(f"• **Recommended Price**: **${res_a.recommended_price:.2f}**")
        st.write(f"• **Assumed Follower**: ${res_a.predicted_follower_price:.2f} (Static)")
        st.write(f"• **Expected Daily Demand**: {res_a.expected_demand:.1f} units")
        st.write(f"• **Expected Daily Profit**: **${res_a.expected_profit:.2f}**")
        st.write(f"• **Unit Margin**: ${res_a.unit_margin:.2f} ({res_a.margin_pct*100:.1f}%)")

    with scol2:
        st.warning("### Strategy B: Game Rule (Match)")
        st.write("Stackelberg leader anticipating follower price matching.")
        st.write(f"• **Recommended Price**: **${res_b.recommended_price:.2f}**")
        st.write(f"• **Predicted Follower**: ${res_b.predicted_follower_price:.2f} (Matched)")
        st.write(f"• **Expected Daily Demand**: {res_b.expected_demand:.1f} units")
        st.write(f"• **Expected Daily Profit**: **${res_b.expected_profit:.2f}**")
        st.write(f"• **Unit Margin**: ${res_b.unit_margin:.2f} ({res_b.margin_pct*100:.1f}%)")

    with scol3:
        st.success("### Strategy C: ML-Aware Stackelberg")
        st.write("Stackelberg leader with ML-predicted follower response.")
        st.write(f"• **Recommended Price**: **${res_c.recommended_price:.2f}**")
        st.write(f"• **Predicted Follower**: ${res_c.predicted_follower_price:.2f} (ML Predicted)")
        st.write(f"• **Expected Daily Demand**: {res_c.expected_demand:.1f} units")
        st.write(f"• **Expected Daily Profit**: **${res_c.expected_profit:.2f}**")
        st.write(f"• **Unit Margin**: ${res_c.unit_margin:.2f} ({res_c.margin_pct*100:.1f}%)")

    st.markdown("---")

    # Realized Multi-Period Simulation Benchmark
    st.subheader("True-World Realized Multi-Period Simulation Benchmark (40 Runs, 28 Days)")

    if not df_summary.empty:
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            fig_bars = plot_strategy_comparison_bars(df_summary)
            st.plotly_chart(fig_bars, use_container_width=True)
        with gcol2:
            if not df_traj.empty:
                fig_cum = plot_cumulative_profit(df_traj)
                st.plotly_chart(fig_cum, use_container_width=True)

        st.dataframe(
            df_summary[[
                "strategy", "mean_total_profit", "ci_95_lower", "ci_95_upper",
                "profit_improvement_pct", "profit_std", "downside_risk_p5", "mean_price"
            ]],
            use_container_width=True,
        )
    else:
        st.info("Benchmark simulation data is being loaded or generated.")

    st.markdown("---")
    st.markdown(
        """
        > **Methodological Honesty Disclaimer**: Walmart M5 data contains **no external competitor prices**.
        > The competitor response model is trained on labeled calibrated simulations to test whether Stackelberg
        > game optimization confers a structural advantage over non-game pricing.
        """
    )
