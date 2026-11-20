"""PricePilot: Strategic Pricing Decision Support System.

Streamlit dashboard entry point.
"""

import streamlit as st


def main() -> None:
    """Run PricePilot Streamlit application."""
    st.set_page_config(
        page_title="PricePilot | Strategic Pricing Decision Support",
        page_icon="📈",
        layout="wide",
    )

    st.title("PricePilot")
    st.subheader("Strategic Pricing Decision Support System")
    st.info(
        "Project skeleton initialized. Pipeline modules are under development."
    )


if __name__ == "__main__":
    main()
