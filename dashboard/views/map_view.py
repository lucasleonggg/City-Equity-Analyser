"""Singapore HDB Access & Affordability Analyser — Map View Page."""

import streamlit as st
import plotly.express as px
import pandas as pd

from data.loader import (
    load_hdb_data,
    get_flat_types,
    get_year_range,
    load_rail_stations,
    get_hospitals_df,
)


def show():
    st.title("Map View")
    st.markdown("Explore HDB resale prices across Singapore's towns.")

    df = load_hdb_data()
    flat_types = get_flat_types()
    yr_min, yr_max = get_year_range()

    with st.sidebar:
        st.header("Map Filters")

        selected_year = st.slider(
            "Year", min_value=yr_min, max_value=yr_max, value=yr_max
        )
        selected_flat_type = st.selectbox("Flat Type", flat_types, index=3)
        show_mrt = st.checkbox("Show Rail Stations (MRT + LRT)", value=True)
        show_hospitals = st.checkbox("Show Hospitals", value=False)

    # Filter to selected year (take Q1 data as representative for the year)
    year_data = df[
        (df["year_int"] == selected_year)
        & (df["flat_type"] == selected_flat_type)
    ].copy()

    # Take latest quarter per town for the selected year
    year_data = year_data.loc[
        year_data.groupby("town")["quarter_sort"].idxmax()
    ].copy()

    if year_data.empty:
        st.info(f"No data for {selected_flat_type} in {selected_year}.")
        return

    # Map
    fig = px.scatter_mapbox(
        year_data,
        lat="lat",
        lon="lng",
        size="median_resale_price",
        color="median_resale_price",
        hover_name="town",
        hover_data={
            "lat": False,
            "lng": False,
            "median_resale_price": ":,.0f",
            "transaction_count": ":,",
            "quarter": True,
        },
        color_continuous_scale="RdYlGn_r",
        range_color=[
            year_data["median_resale_price"].quantile(0.05),
            year_data["median_resale_price"].quantile(0.95),
        ],
        title=f"{selected_flat_type} Resale Prices ({selected_year})",
        mapbox_style="open-street-map",
        center={"lat": 1.3521, "lon": 103.8198},
        zoom=10.5,
        height=600,
    )

    # Add rail stations. Uses load_rail_stations() -- the same authoritative,
    # LRT-inclusive source the equity pages use -- rather than the legacy
    # hardcoded MRT_STATIONS dict, which held only 84 heavy-rail stations and
    # made this page silently disagree with the Equity Map.
    if show_mrt:
        rail_df = load_rail_stations()
        rail_df = rail_df[rail_df["operational"]]
        for mode, colour in (("MRT", "blue"), ("LRT", "purple")):
            subset = rail_df[rail_df["mode"] == mode]
            if subset.empty:
                continue
            fig.add_scattermapbox(
                lat=subset["lat"],
                lon=subset["lng"],
                mode="markers",
                marker=dict(size=6, symbol="circle", color=colour, opacity=0.5),
                name=f"{mode} Stations ({len(subset)})",
                hovertext=subset["station_name"],
                hoverinfo="text",
            )

    # Add hospitals
    if show_hospitals:
        hosp_df = get_hospitals_df()
        fig.add_scattermapbox(
            lat=hosp_df["lat"],
            lon=hosp_df["lng"],
            mode="markers",
            marker=dict(size=10, symbol="hospital", color="red", opacity=0.7),
            name="Hospitals",
            hovertext=hosp_df["hospital"],
            hoverinfo="text",
        )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    # Top 5 most expensive towns
    st.subheader(f"Top 5 Most Expensive Towns ({selected_year}, {selected_flat_type})")
    top5 = year_data.nlargest(5, "median_resale_price")[
        ["town", "median_resale_price", "transaction_count"]
    ]
    st.dataframe(top5, use_container_width=True, hide_index=True)

    # Bottom 5 cheapest towns
    st.subheader(f"Top 5 Most Affordable Towns ({selected_year}, {selected_flat_type})")
    bottom5 = year_data.nsmallest(5, "median_resale_price")[
        ["town", "median_resale_price", "transaction_count"]
    ]
    st.dataframe(bottom5, use_container_width=True, hide_index=True)