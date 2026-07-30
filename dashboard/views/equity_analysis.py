"""Singapore HDB Access & Affordability Analyser — Equity Analysis Page."""

import streamlit as st
from equity.equity_factors import AVG_HOUSEHOLD_SIZE, MONTHS_PER_YEAR
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from data.loader import load_hdb_data, load_income_data, get_town_list, get_flat_types


def show():
    st.title("Affordability Analysis")
    st.markdown(
        "Analyse housing affordability by comparing resale prices to "
        "national-level household income data."
    )

    df = load_hdb_data()
    income_df, median_income_df = load_income_data()
    towns = get_town_list()
    flat_types = get_flat_types()

    st.info(
        "Income data is at the **national level** (not town-level). "
        "Price-to-income ratios provide directional insight rather than "
        "precise town-level affordability."
    )

    # ── Section 1: Price-to-Income Ratio ──
    st.header("Price-to-Income Ratio Over Time")

    col1, col2 = st.columns(2)
    with col1:
        selected_ptowns = st.multiselect(
            "Town", towns, default=["Ang Mo Kio", "Bedok", "Tampines", "Woodlands", "Jurong West"],
            key="pi_towns",
        )
    with col2:
        selected_pft = st.selectbox("Flat Type", flat_types, index=3, key="pi_ft")

    if selected_ptowns and not median_income_df.empty:
        # Compute annual median price by town
        annual_prices = (
            df[(df["flat_type"] == selected_pft) & (df["town"].isin(selected_ptowns))]
            .groupby(["town", "year_int"])["median_resale_price"]
            .mean()
            .reset_index()
        )
        annual_prices = annual_prices.rename(columns={"year_int": "year"})

        # Merge with income data
        merged = annual_prices.merge(
            median_income_df[["year", "income"]], on="year", how="inner"
        )
        # Income series is MONTHLY, PER HOUSEHOLD MEMBER. Annualise (x12) and
        # convert to per-household (x AVG_HOUSEHOLD_SIZE) so this is a real
        # price-to-income multiple and the 5x threshold below is meaningful.
        merged["annual_household_income"] = (
            merged["income"] * MONTHS_PER_YEAR * AVG_HOUSEHOLD_SIZE
        )
        merged["price_to_income"] = (
            merged["median_resale_price"] / merged["annual_household_income"]
        )

        if not merged.empty:
            fig = px.line(
                merged,
                x="year",
                y="price_to_income",
                color="town",
                title=f"Price-to-Income Ratio ({selected_pft})",
                labels={
                    "price_to_income": "Price ÷ annual household income",
                    "year": "Year",
                },
                height=400,
            )
            fig.add_hline(
                y=5, line_dash="dash", line_color="gray",
                annotation_text="5x annual household income",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for the selected filters.")
    else:
        st.info("Select at least one town to display the price-to-income chart.")

    # ── Section 2: Affordability by Town (Latest Year) ──
    st.header("Affordability by Town")
    st.markdown("Median resale price vs. national median income for the latest year.")

    col3, col4 = st.columns(2)
    with col3:
        selected_ayear = st.selectbox(
            "Year",
            sorted(df["year_int"].unique(), reverse=True)[:10],
            index=0,
            key="aff_year",
        )
    with col4:
        selected_aft = st.selectbox("Flat Type", flat_types, index=3, key="aff_ft")

    latest = df[
        (df["year_int"] == selected_ayear)
        & (df["flat_type"] == selected_aft)
    ].copy()

    if not latest.empty and not median_income_df.empty:
        med_income = median_income_df[
            median_income_df["year"] == selected_ayear
        ]["income"].values

        if len(med_income) > 0:
            # Convert monthly per-member -> annual per-household before
            # drawing threshold lines against absolute resale prices.
            med_income = med_income[0] * MONTHS_PER_YEAR * AVG_HOUSEHOLD_SIZE

            # Get latest quarter per town
            latest = latest.loc[
                latest.groupby("town")["quarter_sort"].idxmax()
            ].sort_values("median_resale_price", ascending=False)

            fig = px.bar(
                latest,
                x="town",
                y="median_resale_price",
                title=f"Median Resale Price by Town ({selected_ayear}, {selected_aft})",
                labels={"median_resale_price": "Median Resale Price ($)", "town": "Town"},
                height=400,
                color="median_resale_price",
                color_continuous_scale="RdYlGn_r",
            )
            fig.add_hline(
                y=med_income * 5,
                line_dash="dash",
                line_color="red",
                annotation_text=f"5x annual household income (${med_income * 5:,.0f})",
            )
            fig.add_hline(
                y=med_income * 3,
                line_dash="dot",
                line_color="green",
                annotation_text=f"3x annual household income (${med_income * 3:,.0f})",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # ── Section 3: Income Cross-Section ──
    st.header("Income Cross-Section")
    st.markdown(
        "What income percentile can afford which flat type in a selected town?"
    )

    col5, col6 = st.columns(2)
    with col5:
        selected_ctown = st.selectbox("Town", towns, index=0, key="cs_town")
    with col6:
        selected_cyear = st.selectbox(
            "Year",
            sorted(df["year_int"].unique(), reverse=True)[:10],
            index=0,
            key="cs_year",
        )

    town_data = df[
        (df["town"] == selected_ctown)
        & (df["year_int"] == selected_cyear)
    ].copy()

    if not town_data.empty and not income_df.empty:
        # Get latest quarter data per flat type
        town_data = town_data.loc[
            town_data.groupby("flat_type")["quarter_sort"].idxmax()
        ].sort_values("median_resale_price")

        # Get income data for the year and convert MONTHLY PER-MEMBER figures
        # to ANNUAL PER-HOUSEHOLD, so they are on the same footing as the
        # price-derived "income needed" line. Without this the percentile
        # lines sit at $1k-$9k against a $200k+ requirement and every row
        # reads "N/A (above 90th)".
        year_income = income_df[income_df["year"] == selected_cyear].copy()
        year_income["income"] = (
            year_income["income"] * MONTHS_PER_YEAR * AVG_HOUSEHOLD_SIZE
        )

        # Affordable at a 3x annual-household-income multiple.
        town_data["affordable_income"] = town_data["median_resale_price"] / 3

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=town_data["flat_type"],
            y=town_data["median_resale_price"],
            name="Median Resale Price",
            marker_color="lightblue",
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=town_data["flat_type"],
            y=town_data["affordable_income"],
            name="Annual household income needed (price ÷ 3)",
            marker_color="orange",
            mode="lines+markers",
            yaxis="y",
        ))

        # Add income percentile lines
        for _, row in year_income.iterrows():
            pct = row["percentile"]
            inc = row["income"]
            fig.add_hline(
                y=inc,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"{pct}: ${inc:,.0f}",
            )

        fig.update_layout(
            title=f"Income Cross-Section: {selected_ctown} ({selected_cyear})",
            xaxis_title="Flat Type",
            yaxis_title="Price / annual household income ($)",
            height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.subheader("Affordability Summary")
        affordable_rows = []
        for _, row in town_data.iterrows():
            needed = row["median_resale_price"] / 3
            # Find lowest percentile where income >= needed
            qualifying = year_income[year_income["income"] >= needed]
            if not qualifying.empty:
                min_pct = qualifying.iloc[0]["percentile"]
                affordable_rows.append({
                    "Flat Type": row["flat_type"],
                    "Median Price": f"${row['median_resale_price']:,.0f}",
                    "Annual Household Income Needed": f"${needed:,.0f}",
                    "Min Percentile": min_pct,
                })
            else:
                affordable_rows.append({
                    "Flat Type": row["flat_type"],
                    "Median Price": f"${row['median_resale_price']:,.0f}",
                    "Annual Household Income Needed": f"${needed:,.0f}",
                    "Min Percentile": "Above 90th percentile",
                })

        st.dataframe(
            pd.DataFrame(affordable_rows),
            use_container_width=True,
            hide_index=True,
        )