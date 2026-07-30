"""Singapore HDB Access & Affordability Analyser — Price Trends Page."""

import streamlit as st
import plotly.express as px
import pandas as pd

from data.loader import load_hdb_data, get_town_list, get_flat_types, get_year_range


def show():
    st.title("Price Trends")
    st.markdown("Explore median resale prices across HDB towns and flat types over time.")

    df = load_hdb_data()
    towns = get_town_list()
    flat_types = get_flat_types()
    yr_min, yr_max = get_year_range()

    # Filters sidebar
    with st.sidebar:
        st.header("Filters")

        selected_towns = st.multiselect(
            "Town", towns, default=towns[:5]
        )
        selected_flat_types = st.multiselect(
            "Flat Type", flat_types, default=["4-Room"]
        )
        year_range = st.slider(
            "Year Range",
            min_value=yr_min,
            max_value=yr_max,
            value=(max(yr_min, yr_max - 10), yr_max),
        )
        price_metric = st.radio("Price Metric", ["median_resale_price", "mean_resale_price"], index=0)
        metric_label = "Median Price" if price_metric == "median_resale_price" else "Mean Price"

    # Filter data
    mask = (
        df["town"].isin(selected_towns)
        & df["flat_type"].isin(selected_flat_types)
        & (df["year_int"] >= year_range[0])
        & (df["year_int"] <= year_range[1])
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.info("No data matches the selected filters.")
        return

    # Sort chronologically
    filtered = filtered.sort_values("quarter_sort")

    # Line chart
    fig = px.line(
        filtered,
        x="quarter",
        y=price_metric,
        color="town",
        line_group="flat_type",
        symbol="flat_type",
        title=f"{metric_label} by Town and Flat Type",
        labels={price_metric: metric_label, "quarter": "Quarter"},
        height=500,
    )
    fig.update_layout(legend_title_text="Town")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Towns Selected", len(selected_towns))
    with col2:
        st.metric("Flat Types Selected", len(selected_flat_types))
    with col3:
        avg_price = filtered[price_metric].mean()
        st.metric(f"Avg {metric_label}", f"${avg_price:,.0f}")

    # Data table
    with st.expander("View Raw Data"):
        display_cols = [
            "town", "quarter", "flat_type", price_metric,
            "transaction_count", "median_floor_area_sqm",
        ]
        st.dataframe(
            filtered[display_cols].sort_values(["town", "quarter"]),
            use_container_width=True,
            hide_index=True,
        )