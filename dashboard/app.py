"""Singapore HDB Access & Affordability Analyser — Streamlit Dashboard.

Entry point with page navigation.
"""

import streamlit as st
from views import price_trends, map_view, equity_analysis, equity_map, transaction_model

st.set_page_config(
    page_title="Singapore HDB Access & Affordability Analyser",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Singapore HDB Access & Affordability Analyser")
st.sidebar.markdown(
    "Analyse HDB resale price trends across Singapore's towns. "
    "Data from Data.gov.sg covering 1990–2026."
)

page = st.sidebar.radio(
    "Navigate",
    ["Price Trends", "Map View", "Affordability Analysis",
     "Access & Affordability Map", "Transaction-Level Price Model"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Streamlit + Plotly | "
    "[Data.gov.sg](https://data.gov.sg)"
)

if page == "Price Trends":
    price_trends.show()
elif page == "Map View":
    map_view.show()
elif page == "Affordability Analysis":
    equity_analysis.show()
elif page == "Access & Affordability Map":
    equity_map.show()
elif page == "Transaction-Level Price Model":
    transaction_model.show()