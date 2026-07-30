"""Regression tests for the equity pipeline.

Each test here exists because something was actually wrong, or because an
invariant is easy to break silently. Run with:

    pip install pytest
    python -m pytest tests/ -v

from the project root.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"))

from data.loader import (  # noqa: E402
    load_rail_stations,
    load_hospitals_raw,
    load_polyclinics_raw,
    load_income_data,
)
from equity.equity_factors import (  # noqa: E402
    annualised_household_income,
    compute_equity_scores,
    compute_factor_details,
    compute_getis_ord,
    compute_weight_sensitivity,
    _real_town_features,
)


# --------------------------------------------------------------------------
# Affordability was dimensionally wrong: a resale price was divided by a
# MONTHLY per-member income, producing "price-to-income ratios" of 156-289
# and a nonsensical "5x median income" threshold line.
# --------------------------------------------------------------------------

def test_household_income_is_annualised_not_monthly():
    from equity.equity_factors import AVG_HOUSEHOLD_SIZE, MONTHS_PER_YEAR

    annual, source_year, monthly_per_member = annualised_household_income()
    # The source series is MONTHLY income per household MEMBER. Both
    # corrections must be applied: x12, and x average household size.
    assert annual == pytest.approx(
        monthly_per_member * MONTHS_PER_YEAR * AVG_HOUSEHOLD_SIZE, rel=1e-6
    )
    assert annual > monthly_per_member * 12, "household size correction not applied"
    # Sanity band for Singapore annual household income from work.
    assert 60_000 < annual < 400_000, f"implausible annual household income: {annual}"
    assert 2000 <= source_year <= 2030


def test_price_to_income_ratios_are_plausible():
    """A price-to-income ratio should land in single/low-double digits, not
    the 156-289 range the monthly-income bug produced."""
    df = _real_town_features()
    ratios = df["price_to_income_ratio"].dropna()
    assert len(ratios) > 0
    assert ratios.max() < 40, f"max ratio {ratios.max():.1f} suggests a units error"
    assert ratios.min() > 1, f"min ratio {ratios.min():.1f} suggests a units error"


# --------------------------------------------------------------------------
# The transit factor silently excluded every LRT station, which gutted
# Punggol / Sengkang / Bukit Panjang -- exactly the towns whose feeder loops
# make them accessible.
# --------------------------------------------------------------------------

def test_rail_stations_include_lrt():
    rail = load_rail_stations()
    modes = set(rail["mode"].unique())
    assert "LRT" in modes, "LRT stations missing -- transit factor will be biased"
    assert (rail["mode"] == "LRT").sum() >= 30, "suspiciously few LRT stations"


def test_lrt_towns_are_not_starved_of_stations():
    """Punggol and Sengkang have LRT loops; they should not score 1-2 stations."""
    df = _real_town_features().set_index("town")
    for town in ("Punggol", "Sengkang"):
        if town in df.index:
            assert df.loc[town, "station_count"] >= 5, (
                f"{town} has only {df.loc[town, 'station_count']} stations -- "
                "LRT loop likely dropped again"
            )


# --------------------------------------------------------------------------
# Jurong East's town centroid is byte-identical to a JOB_CENTERS entry, so its
# commute distance is exactly zero. That must stay flagged, not silently
# become a "best commute in Singapore" result.
# --------------------------------------------------------------------------

def test_degenerate_commute_is_flagged():
    df = _real_town_features()
    assert "commute_is_degenerate" in df.columns
    flagged = set(df.loc[df["commute_is_degenerate"], "town"])
    assert "Jurong East" in flagged, (
        "Jurong East sits on a job centre; its zero commute distance must be flagged"
    )


# --------------------------------------------------------------------------
# Composite score integrity.
# --------------------------------------------------------------------------

def test_no_town_has_nan_composite():
    df = compute_equity_scores()
    assert df["composite_score"].notna().all(), "NaN composite score present"


def test_all_towns_retained_in_composite():
    """Towns were previously dropped for missing estate age, silently removing
    Central Area / Bukit Timah / Kallang-Whampoa from the equity map."""
    df = compute_equity_scores()
    for town in ("Central Area", "Bukit Timah", "Kallang/Whampoa"):
        assert town in set(df["town"]), f"{town} dropped from the composite"


def test_normalised_factors_are_in_unit_range():
    df = compute_equity_scores()
    for col in ("affordability", "transit_access", "healthcare_access",
                "commute_access", "estate_modernity", "composite_score"):
        assert df[col].between(0, 1).all(), f"{col} outside [0, 1]"


# --------------------------------------------------------------------------
# Getis-Ord: permutation inference, FDR correction.
# --------------------------------------------------------------------------

def test_getis_ord_pseudo_pvalues_are_valid():
    gi = compute_getis_ord(n_permutations=999)
    assert gi["p_sim"].between(0, 1).all()
    assert gi["p_sim_fdr"].between(0, 1).all()
    # Pseudo p can never be exactly 0 under the (r+1)/(m+1) convention.
    assert (gi["p_sim"] > 0).all()
    # FDR correction can only make p-values larger (or equal).
    assert (gi["p_sim_fdr"] >= gi["p_sim"] - 1e-9).all()


def test_getis_ord_significance_is_conservative():
    """FDR correction should not mark every town as significant -- if it does,
    something has gone wrong with the correction."""
    gi = compute_getis_ord(n_permutations=999)
    assert gi["significant_fdr"].sum() < len(gi), "all towns flagged significant"


def test_getis_ord_is_deterministic():
    a = compute_getis_ord(n_permutations=999, random_state=7)
    b = compute_getis_ord(n_permutations=999, random_state=7)
    np.testing.assert_allclose(a["gi_star_z"], b["gi_star_z"])


# --------------------------------------------------------------------------
# Weighting is an assertion, not a finding -- sensitivity analysis must exist
# and must show the ranking is not wildly weight-dependent.
# --------------------------------------------------------------------------

def test_weight_sensitivity_reports_rank_correlation():
    sens = compute_weight_sensitivity()
    assert len(sens) >= 2, "need at least one alternative weighting to compare"
    corr_col = [c for c in sens.columns if "spearman" in c.lower()]
    assert corr_col, f"no Spearman column found in {list(sens.columns)}"
    vals = sens[corr_col[0]].dropna()
    assert vals.between(-1, 1).all()


# --------------------------------------------------------------------------
# Healthcare source integrity.
# --------------------------------------------------------------------------

def test_all_three_polyclinic_clusters_present():
    poly = load_polyclinics_raw()
    operators = set(poly["operator"].unique())
    assert {"SingHealth", "NHG", "NUHS"}.issubset(operators), (
        f"missing a polyclinic cluster; found {operators}"
    )
    assert len(poly) == 28, f"expected 28 polyclinics, got {len(poly)}"


def test_no_duplicate_polyclinics():
    poly = load_polyclinics_raw()
    assert not poly["polyclinic"].duplicated().any(), "duplicate polyclinic rows"


def test_every_hospital_has_a_town():
    hosp = load_hospitals_raw()
    assert hosp["town"].notna().all()
    assert len(hosp) == 28, f"expected 28 operational hospitals, got {len(hosp)}"


# --------------------------------------------------------------------------
# Factor details must agree with the underlying feature table.
# --------------------------------------------------------------------------

def test_factor_details_match_feature_table():
    df = _real_town_features().set_index("town")
    town = "Woodlands"
    details = compute_factor_details(town)
    assert details is not None
    assert details["rail_stations"] == int(df.loc[town, "station_count"])
    assert details["polyclinics"] == int(df.loc[town, "polyclinic_count"])
    assert details["population"] == int(df.loc[town, "population"])


def test_healthcare_uses_distance_not_boundary_count():
    """Counting facilities inside a town boundary gave Bedok 0 hospitals
    because Changi General fell just outside it. Distance-to-nearest is the
    defensible measure, so every town must have a finite one."""
    df = _real_town_features()
    assert "nearest_hospital_km" in df.columns
    assert df["nearest_hospital_km"].notna().all()
    assert (df["nearest_hospital_km"] > 0).all()
    # No town in Singapore is more than ~25km from a hospital.
    assert df["nearest_hospital_km"].max() < 25


def test_lrt_breakdown_is_reported_and_consistent():
    """`of_which_lrt` must never exceed the total station count."""
    for town in ("Punggol", "Sengkang", "Bukit Panjang", "Woodlands"):
        details = compute_factor_details(town)
        if details is None:
            continue
        assert 0 <= details["of_which_lrt"] <= details["rail_stations"]


def test_factor_details_unknown_town_returns_none():
    assert compute_factor_details("Atlantis") is None
