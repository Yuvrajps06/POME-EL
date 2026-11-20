"""PricePilot: Strategic Pricing Decision Support System.

Main application entrypoint and multi-page Streamlit navigation.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import (
    overview,
    market,
    demand,
    competitor,
    optimizer,
    comparison,
    sensitivity,
)


def main() -> None:
    """Run PricePilot Streamlit application with multi-page navigation."""
    st.set_page_config(
        page_title="PricePilot | Strategic Pricing Support",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar Navigation
    st.sidebar.title("✈️ PricePilot")
    st.sidebar.caption("Strategic Pricing Decision Support System")

    pages = {
        "📊 Executive Summary": overview.render,
        "📈 Market Intelligence": market.render,
        "📉 Demand Forecasting": demand.render,
        "🎯 Competitor Intelligence": competitor.render,
        "⚙️ Price Optimizer": optimizer.render,
        "⚖️ Strategy Comparison": comparison.render,
        "🔬 Sensitivity Analysis": sensitivity.render,
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specs**:
        - Data: Walmart M5 Retail
        - Stores: CA_1 (Leader), CA_2 (Competitor)
        - Category: FOODS_3
        - Theory: Stackelberg Game Theory
        """
    )
    st.sidebar.caption("v1.0.0 | Antigravity AI")

    # Render selected page
    render_func = pages[selected_page_name]
    render_func()


if __name__ == "__main__":
    main()
