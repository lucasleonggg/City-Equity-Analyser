"""Access & Affordability Map — heatmap of the composite score with factor
breakdown and clustering.

NAMING: the score is a weighted mean of amenity and affordability factors. It
is deliberately NOT called an equity or deprivation index -- see the "what the
composite is and is not" section of README.md. The module and package are
still named `equity_*` for import stability; the user-facing labels are not.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from equity.equity_factors import (
    compute_equity_scores,
    compute_factor_details,
    compute_getis_ord,
    compute_shapley_price_drivers,
    compute_weight_sensitivity,
    INTEL_ACCELERATION,
)


FACTOR_LABELS = {
    "affordability": "Affordability",
    "transit_access": "Transit Access",
    "healthcare_access": "Healthcare Access",
    "commute_access": "Commute Access",
    "estate_modernity": "Estate Modernity",
}

CLUSTER_LABELS = {
    0: "Underserved (all factors low)",
    1: "Transit-poor but affordable",
    2: "Balanced middle",
    3: "Well-served (all factors high)",
}

CLUSTER_COLORS = {0: "#d73027", 1: "#fc8d59", 2: "#91bfdb", 3: "#1a9641"}


def show():
    st.title("Access & Affordability Heatmap")
    if INTEL_ACCELERATION:
        st.caption("Clustering and price modeling accelerated via Intel(R) Extension for Scikit-learn (oneDAL backend).")
    else:
        st.caption(
            "Running on stock scikit-learn -- Intel(R) Extension for Scikit-learn "
            "(`pip install scikit-learn-intelex`) is not installed in this environment."
        )
    st.markdown(
        "Composite access-and-affordability score across five dimensions. "
        "**Dark red** areas score lowest across the combined factors."
    )

    df = compute_equity_scores()

    # ── Map ──
    fig = px.scatter_mapbox(
        df,
        lat="lat", lon="lng",
        size="composite_score",
        color="composite_score",
        hover_name="town",
        hover_data={
            "lat": False, "lng": False,
            "composite_score": ":.3f",
            "affordability": ":.2f",
            "transit_access": ":.2f",
            "healthcare_access": ":.2f",
            "commute_access": ":.2f",
            "estate_modernity": ":.2f",
            "cluster": True,
        },
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="Composite Access & Affordability Score by Town",
        mapbox_style="open-street-map",
        center={"lat": 1.35, "lon": 103.82},
        zoom=10.3,
        height=600,
        size_max=30,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Dark red = low composite score (least equitable). "
        "Dark green = high composite score (most equitable). "
        "Hover for factor breakdown."
    )

    # ── Selected town detail ──
    towns = sorted(df["town"].unique())
    selected = st.selectbox("Select a town for detailed factor breakdown", towns)

    if selected:
        details = compute_factor_details(selected)
        row = df[df["town"] == selected].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Composite Score", f"{row['composite_score']:.3f}")
        col2.metric("Cluster", CLUSTER_LABELS.get(row["cluster"], f"Cluster {row['cluster']}"))
        col3.metric("Population", f"{details['population']:,}")

        # Radar chart
        factor_cols = list(FACTOR_LABELS.keys())
        values = [row[c] for c in factor_cols]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=list(FACTOR_LABELS.values()) + [list(FACTOR_LABELS.values())[0]],
            fill="toself",
            name=selected,
        ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{selected} — Factor Scores (normalized)",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Raw values table
        st.subheader("Raw Values")
        st.table(pd.DataFrame([
            {"Factor": "Price ÷ annual household income", "Value": details["price_to_annual_household_income"]},
            {"Factor": "Rail stations (nearest-town assigned)", "Value": details["rail_stations"]},
            {"Factor": "  of which LRT", "Value": details["of_which_lrt"]},
            {"Factor": "Distance to nearest hospital (km)", "Value": details["nearest_hospital_km"]},
            {"Factor": "Polyclinics in town (all 3 clusters)", "Value": details["polyclinics"]},
            {"Factor": "Avg Estate Age (years)", "Value": details["estate_age_years"]},
            {"Factor": "Commute Proxy (min, see caveat)", "Value": f'{details["commute_minutes"]:.1f}'},
            {"Factor": "Population (HDB, 31 Mar 2024)", "Value": f"{details['population']:,}"},
        ]).set_index("Factor"))
        st.caption(
            "Transit counts reflect only stations with known coordinates in this "
            "build (partial coverage -- see equity_factors.py docstring). Commute "
            "is a distance-based proxy, not measured journey time."
        )

    # ── Getis-Ord Gi* hotspot analysis ──
    st.header("Spatial Hotspots (Getis-Ord Gi*)")
    st.markdown(
        "Identifies spatial clusters of high or low composite scores among each "
        "town's 5 nearest neighbours, rather than ranking towns individually. "
        "Significance comes from **conditional permutation inference** "
        "(9,999 reshuffles per town), not a normal approximation, and is "
        "**Benjamini-Hochberg FDR-corrected** across all 26 towns."
    )
    gi_df = compute_getis_ord()
    fig_gi = px.bar(
        gi_df, x="town", y="gi_star_z", color="cluster_label",
        title="Getis-Ord Gi* z-score by town (colour = FDR-corrected significance)",
        color_discrete_map={
            "High cluster (FDR<0.05)": "#1a9641",
            "No clustering": "#cccccc",
            "Low cluster (FDR<0.05)": "#d73027",
        },
        height=400,
    )
    fig_gi.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_gi, use_container_width=True)

    n_sig = int(gi_df["significant_fdr"].sum())
    st.caption(
        f"Only **{n_sig} of {len(gi_df)}** towns remain significant after FDR "
        "correction. Under the weaker normal-approximation bands used before, "
        "several more towns appeared to cluster -- permutation inference does "
        "not support those. Note also that a LOW composite here means "
        "'expensive with older stock' (mature central estates), not 'deprived'."
    )
    with st.expander("Full Gi* table with pseudo p-values"):
        st.dataframe(
            gi_df[["town", "composite_score", "gi_star_z", "p_sim", "p_sim_fdr", "significant_fdr"]],
            use_container_width=True, hide_index=True,
        )

    # ── Weight sensitivity ──
    st.header("How Much Does the Equal Weighting Matter?")
    st.markdown(
        "The composite is an unweighted mean by default. That is a choice, not "
        "a finding, so here is how the ranking moves under alternative "
        "weightings (Spearman rank correlation vs. the default)."
    )
    st.dataframe(compute_weight_sensitivity(), use_container_width=True, hide_index=True)

    # ── Shapley-value price driver breakdown ──
    st.header("What Drives Resale Price? (Shapley Decomposition)")
    st.markdown(
        "Exact Shapley attribution against the TRANSACTION-LEVEL model "
        "(982k real resale transactions), for a representative recent flat in "
        "the selected town. Features: floor area, storey, remaining lease, "
        "year, town price rank, and flat size. Shows what pushes that flat's "
        "predicted price above or below the all-towns baseline."
    )
    shap_town = st.selectbox("Select a town for price driver breakdown", towns, key="shap_town")
    shap_result = compute_shapley_price_drivers(shap_town)
    if shap_result is None:
        st.info(
            "Shapley model not trained yet. Run "
            "`python pipeline/train_transaction_model.py` to build it."
        )
    else:
        st.caption(
            f"Attribution against a model fit on "
            f"{shap_result['n_training_rows']:,} real transactions "
            "(previously this explained a forest fit on just 23 town rows)."
        )
        contrib_df = pd.DataFrame([
            {"Feature": k, "Contribution ($)": v}
            for k, v in shap_result["contributions"].items()
        ]).sort_values("Contribution ($)")
        fig_shap = px.bar(
            contrib_df, x="Contribution ($)", y="Feature", orientation="h",
            title=f"{shap_town}: baseline ${shap_result['baseline_price']:,.0f} "
                  f"-> predicted ${shap_result['predicted_price']:,.0f}",
            color="Contribution ($)", color_continuous_scale="RdYlGn",
            height=350,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # ── Factor correlation heatmap ──
    st.header("Factor Correlation")
    corr = df[factor_cols].corr()
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlBu_r",
        title="Factor Correlation Matrix",
        height=500,
        labels={"x": "Factor", "y": "Factor"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Clustering ──
    st.header("Town Clusters")
    st.markdown("k-means clustering groups towns by their access-and-affordability profile.")

    cluster_df = df[["town", "composite_score", "cluster"] + factor_cols].copy()
    cluster_df["cluster_label"] = cluster_df["cluster"].map(CLUSTER_LABELS)
    cluster_df = cluster_df.sort_values(["cluster", "composite_score"], ascending=[True, False])

    # Color the cluster column
    def color_cluster(val):
        color = CLUSTER_COLORS.get(val, "#ccc")
        return f"background-color: {color}; color: {'white' if val in (0, 3) else 'black'}"

    # Color the cluster column.
    # `Styler.applymap` was deprecated in pandas 2.1 and REMOVED in pandas 3.0,
    # renamed to `Styler.map`. requirements.txt pins `pandas>=2.0`, so a fresh
    # deployment resolves to 3.x and the old name raises AttributeError at
    # render time -- a crash that only appears on a clean install, never in a
    # dev environment with an older pandas already present.
    styler = cluster_df.style
    _map = getattr(styler, "map", None) or styler.applymap
    styled = _map(color_cluster, subset=["cluster"]) \
        .format({c: "{:.3f}" for c in factor_cols + ["composite_score"]})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Cluster bar chart ──
    avg_cluster = cluster_df.groupby("cluster")[factor_cols].mean().reset_index()
    avg_cluster["cluster_label"] = avg_cluster["cluster"].map(CLUSTER_LABELS)
    fig4 = go.Figure()
    for _, row in avg_cluster.iterrows():
        fig4.add_trace(go.Bar(
            name=row["cluster_label"],
            x=list(FACTOR_LABELS.values()),
            y=[row[c] for c in factor_cols],
            marker_color=CLUSTER_COLORS.get(row["cluster"], "#ccc"),
        ))
    fig4.update_layout(
        title="Average Factor Scores by Cluster",
        barmode="group",
        height=450,
    )
    st.plotly_chart(fig4, use_container_width=True)