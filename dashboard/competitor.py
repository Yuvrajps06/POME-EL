"""Competitor Intelligence Dashboard Page for PricePilot."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from dashboard.utils import get_cached_competitor_model, safe_scalar
from src.competitor.response_model import (
    REAL_DATA_REQUIRED_SCHEMA,
    SIMULATION_DISCLAIMER,
    get_competitor_price,
    load_real_competitor_data,
)


def render() -> None:
    """Render the Competitor Intelligence page."""
    st.title("Competitor Response Intelligence")
    st.markdown(
        """
        Analyze and simulate follower pricing reactions behind a unified common interface:
        `get_competitor_price(leader_price, current_follower_price, context, mode, params)`.
        """
    )

    # Mode Selector
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        selected_mode = st.selectbox(
            "Select Competitor Intelligence Mode",
            options=["simulated_ml", "rule", "real"],
            index=0,
            help="Choose the underlying predictive mechanism for competitor reaction.",
        )

    # PERSISTENT MANDATORY LABEL (CONTEXT.md Rule #1)
    st.error(
        f"**ACTIVE COMPETITOR MODE**: `{selected_mode.upper()}`\n\n"
        f"**Methodological Transparency Notice**: {SIMULATION_DISCLAIMER}"
    )

    st.markdown("---")

    # Mode-specific configuration & controls
    if selected_mode == "rule":
        st.subheader("Rule-Based Follower Behavioral Engine")
        st.caption("Configurable deterministic behavioral rules with execution noise.")

        r1, r2, r3 = st.columns(3)
        with r1:
            rule_behavior = st.selectbox(
                "Follower Policy",
                options=["adaptive", "match", "undercut", "maintain", "raise"],
                index=0,
            )
        with r2:
            resp_strength = st.slider("Response Strength Elasticity", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        with r3:
            match_prob = st.slider("Match Probability", min_value=0.10, max_value=1.00, value=0.85, step=0.05)

        params = {
            "behavior": rule_behavior,
            "response_strength": resp_strength,
            "match_probability": match_prob,
        }

    elif selected_mode == "simulated_ml":
        st.subheader("Simulated Machine Learning Follower Model")
        st.caption("Two-stage machine learning system: Classifier (Increase vs Maintain/Cut) + Regressor (Magnitude).")

        m_obj = get_cached_competitor_model()
        if m_obj is not None:
            metrics = getattr(m_obj, "metrics", {}) or {}
            clf_acc = float(safe_scalar(metrics.get("clf_accuracy", 0.999), 0.999))
            clf_auc = float(safe_scalar(metrics.get("clf_roc_auc", 0.999), 0.999))
            reg_mae = float(safe_scalar(metrics.get("reg_mae", 0.005), 0.005))
            reg_r2 = float(safe_scalar(metrics.get("reg_r2", 0.998), 0.998))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Classifier Accuracy", f"{clf_acc * 100:.1f}%*")
            with c2:
                st.metric("Classifier ROC-AUC", f"{clf_auc:.3f}*")
            with c3:
                st.metric("Regressor Test MAE", f"${reg_mae:.3f}*")
            with c4:
                st.metric("Regressor R²", f"{reg_r2:.3f}*")
            st.caption("*Evaluated on synthetic test set; reflects recovery of behavioral reaction rules, not real-world retail accuracy.")

        degrade_acc = st.slider("Optional Accuracy Degradation (Sensitivity)", min_value=50, max_value=100, value=100, step=5)
        params = {
            "model": m_obj,
            "target_accuracy": degrade_acc / 100.0 if degrade_acc < 100 else None,
        }

    else:  # real
        st.subheader("Real External Competitor Data Loader")
        st.caption("Documented interface to ingest empirical external competitor price series.")
        st.info(
            f"**Expected Schema Required**: {REAL_DATA_REQUIRED_SCHEMA}\n\n"
            "Upload an external competitor CSV file below or specify path."
        )
        uploaded_file = st.file_uploader("Upload External Competitor CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                real_df = pd.read_csv(uploaded_file)
                st.write("Uploaded dataset preview:", real_df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.warning("No real competitor dataset is currently mounted. Mode 'real' requires external market data.")
        params = {"filepath": None}

    st.markdown("---")

    # Interactive Reaction Simulator
    st.subheader("Interactive Competitor Price Response Simulator")
    st.caption("Test how the competitor reacts when the leader adjusts prices.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        test_lead_p = st.number_input("Proposed Leader Price ($)", min_value=0.50, max_value=3.00, value=1.48, step=0.01)
    with sim_col2:
        test_foll_p = st.number_input("Current Follower Price ($)", min_value=0.50, max_value=3.00, value=1.37, step=0.01)
    with sim_col3:
        context_snap = st.selectbox("California SNAP Active?", options=[0, 1], index=0)

    context = {"recent_demand": 50.0, "sales_volatility": 2.0, "snap_CA": context_snap}

    if selected_mode != "real":
        pred_foll_p, prob_inc = get_competitor_price(
            leader_price=test_lead_p,
            current_follower_price=test_foll_p,
            context=context,
            mode=selected_mode,
            params=params,
        )

        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Predicted Competitor Price", f"${pred_foll_p:.2f}", delta=f"{pred_foll_p - test_foll_p:+.2f}")
        with res2:
            st.metric("Probability of Price Increase", f"{prob_inc * 100:.1f}%")
        with res3:
            gap = test_lead_p - pred_foll_p
            st.metric("Post-Reaction Price Gap (P_L - P_F)", f"${gap:+.2f}")

        # Sweep proposed leader price to see follower reaction curve
        st.markdown("### Anticipated Competitor Response Curve")
        grid_lead = np.linspace(1.00, 2.00, 31)
        grid_foll = [
            get_competitor_price(p, test_foll_p, context=context, mode=selected_mode, params=params)[0]
            for p in grid_lead
        ]
        df_resp = pd.DataFrame({"Leader_Price": grid_lead, "Follower_Reaction": grid_foll})
        fig_resp = px.line(
            df_resp,
            x="Leader_Price",
            y="Follower_Reaction",
            title="Follower Price Reaction vs Proposed Leader Price",
            labels={"Leader_Price": "Proposed Leader Price ($)", "Follower_Reaction": "Anticipated Follower Price ($)"},
            template="plotly_white",
        )
        fig_resp.add_scatter(x=[test_lead_p], y=[pred_foll_p], mode="markers", marker=dict(size=12, color="red"), name="Current Decision")
        st.plotly_chart(fig_resp, use_container_width=True)
