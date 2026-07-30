"""Town-level housing-access factors, computed from real data only.

NAMING NOTE (read this before citing any output)
-------------------------------------------------
The composite below is deliberately NOT called an "equity index". It is an
unweighted-by-default average of five *amenity and affordability* factors.
High score = cheaper relative to income, newer housing stock, more rail
stations per capita, closer to a hospital, shorter distance to a job centre.

That is a measure of "how well-served and affordable is this town", which is
NOT the same thing as "how inequitable is this town". A mature central estate
(Bishan, Ang Mo Kio, Bukit Merah) scores LOW here mainly because it is
expensive and its flats are old -- not because its residents are underserved.
Read the composite as *access-and-affordability*, and read a low score as
"expensive, older stock", not as "deprived". The `compute_weight_sensitivity`
function exists so this equal weighting can be challenged rather than
asserted.

FACTOR PROVENANCE
------------------
- affordability      REAL. Town median resale price (latest year, averaged
                     across flat types) / ANNUALISED household income.
                     See the dimensional note below -- this was previously
                     wrong by a factor of 12 and by a per-member/per-household
                     mixup.
- transit_access     REAL. Rail stations per 10k residents, from LTA's exit
                     GEOJSON, INCLUDING LRT. Stations are assigned to their
                     nearest town centroid rather than counted inside a fixed
                     1.5km buffer, so feeder loops (Punggol, Sengkang, Bukit
                     Panjang) are captured instead of being cut off by an
                     arbitrary radius.
- healthcare_access  REAL. Distance from town centroid to the NEAREST hospital
                     (all 28 have real coordinates) combined with polyclinics
                     per capita. Distance-based rather than
                     count-inside-boundary, because the old version scored
                     Bedok at 0 hospitals purely because Changi General
                     happened to be assigned to Tampines.
- commute_access     PROXY, deterministic. Straight-line distance from town
                     centroid to the nearest of four job centres, converted to
                     minutes at a fixed assumed speed. NOT measured journey
                     time. See the Jurong East caveat below.
- estate_modernity   REAL. Average estate age from estate-maturity.csv.
                     Interpret as "less near-term upgrading need", not as a
                     livability judgment.
- education_access,
  green_space        REMOVED, not fabricated. No school-location or
                     green-space dataset exists in raw_data/. Add them back
                     only against a real source (MOE school directory,
                     NParks green space layer).

DIMENSIONAL NOTE ON AFFORDABILITY (previously a real bug)
----------------------------------------------------------
The income CSV is "Monthly Household Employment Income Per Household Member".
The old code divided a resale PRICE by that raw MONTHLY PER-MEMBER figure,
producing ratios of 156-289 that are not price-to-income multiples in any
recognised sense, and which made the "5x threshold" annotations in
equity_analysis.py meaningless. Two corrections are applied here:
  1. x12 to annualise.
  2. x AVG_HOUSEHOLD_SIZE to convert per-member to per-household, since the
     conventional "median multiple" (and the 5x affordability threshold) is
     defined against total household income.
Correction (2) depends on an assumed household size, which is stated
explicitly as a constant rather than buried, and is the one number in this
factor that is an assumption rather than a direct reading.

Also included here, because they were claimed as this project's methodology
but did not exist in the original codebase:
- Getis-Ord Gi* hotspot statistics (implemented directly in numpy; esda /
  libpysal are not installable in the offline build environment).
- Shapley attribution. NOTE: this now explains the TRANSACTION-LEVEL model
  fit on 982,011 rows (see pipeline/train_transaction_model.py), not a
  RandomForest fit on 23 town rows. Explaining a 23-row forest was explaining
  noise.
"""

import itertools
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd

# --- Intel(R) Extension for Scikit-learn -----------------------------------
# Patches scikit-learn's KMeans / RandomForestRegressor / MinMaxScaler onto
# Intel's oneDAL backend. Applied here for consistency, but be honest about
# scale: the town table is 23 rows, far too small for a backend swap to
# matter. The dataset where oneDAL can actually show something is the
# 982,011-row transaction model in pipeline/train_transaction_model.py, which
# carries a measured stock-scikit-learn baseline and an explicitly
# NOT-YET-MEASURED accelerated number.
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_ACCELERATION = True
except ImportError:
    INTEL_ACCELERATION = False
# ---------------------------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from data.loader import (
    TOWN_COORDS,
    load_hdb_data,
    load_income_data,
    load_rail_stations,
    load_hospitals_raw,
    load_polyclinics_raw,
    load_estate_maturity_raw,
)

# HDB Annual Report 2023/2024, "Resident Population by Town as at 31 March
# 2024" (source: Singapore Department of Statistics). Kallang/Whampoa is
# DERIVED by subtraction from the published total (the source table cut it
# off), so it is the least certain entry here. Lim Chu Kang has no current
# figure -- a defunct/rural estate with only historical resale transactions.
TOWN_POPULATION_2024 = {
    "Ang Mo Kio": 127750, "Bedok": 171110, "Bishan": 56790,
    "Bukit Batok": 132230, "Bukit Merah": 128000, "Bukit Panjang": 114460,
    "Choa Chu Kang": 163610, "Clementi": 71140, "Geylang": 84430,
    "Hougang": 165450, "Jurong East": 67640, "Jurong West": 234940,
    "Kallang/Whampoa": 97240,  # derived by subtraction
    "Pasir Ris": 98470, "Punggol": 171290, "Queenstown": 78400,
    "Sembawang": 92090, "Sengkang": 224310, "Serangoon": 59970,
    "Tampines": 242610, "Toa Payoh": 112660, "Woodlands": 235690,
    "Yishun": 201850, "Bukit Timah": 7110, "Central Area": 25490,
    "Marine Parade": 18360,
    "Lim Chu Kang": None,
}

# ASSUMPTION, not a reading from the data. Average Singapore resident
# household size, used only to convert the per-household-MEMBER income series
# into an approximate per-HOUSEHOLD figure so the conventional 5x
# "median multiple" affordability threshold is meaningful. Sensitivity: the
# affordability RANKING is unaffected by this constant (it scales every town
# identically and is then min-max normalised); only the displayed ratio and
# the 5x threshold comparison move.
AVG_HOUSEHOLD_SIZE = 3.09
MONTHS_PER_YEAR = 12

# Towns whose estate-age row is filed under a different name in
# estate-maturity.csv. Verified by inspection of that file's town column.
ESTATE_NAME_ALIASES = {
    "Kallang/Whampoa": "Kallang",
    "Central Area": "Pearl's Hill (Chinatown)",
}

JOB_CENTERS = {
    "Raffles Place (CBD)": (1.2837, 103.8510),
    "Jurong East Regional Centre": (1.3329, 103.7414),
    "Tampines Regional Centre": (1.3523, 103.9448),
    "one-north": (1.2994, 103.7869),
}

# Assumed effective door-to-door speed (walk + wait + in-vehicle), used only
# to turn straight-line distance into rough minutes. A LABELLED PROXY, not
# measured LTA journey-planner time.
ASSUMED_SPEED_KMH = 28
FIXED_OVERHEAD_MIN = 12

# Default composite weights. Equal by default, but explicit and overridable
# so the weighting is a stated choice rather than a silent assumption. See
# compute_weight_sensitivity() for how much the ranking depends on it.
FACTOR_COLUMNS = [
    "affordability", "transit_access", "healthcare_access",
    "commute_access", "estate_modernity",
]
DEFAULT_WEIGHTS = {c: 1.0 for c in FACTOR_COLUMNS}


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _commute_minutes(lat, lon):
    """Deterministic distance-based commute proxy.

    KNOWN DEGENERACY: TOWN_COORDS['Jurong East'] is byte-identical to
    JOB_CENTERS['Jurong East Regional Centre'], so its distance is exactly 0
    and its commute collapses to FIXED_OVERHEAD_MIN (12.0) -- a circular
    result rather than an independent measurement. Tampines is close to the
    same issue. Both are flagged in commute_is_degenerate on the feature
    table rather than quietly left to look like findings. Replacing this
    proxy with real LTA journey-planner times would remove the problem
    entirely; until then, treat Jurong East's commute score as an artifact.
    """
    dist_km = min(
        _haversine(lat, lon, jlat, jlon) for jlat, jlon in JOB_CENTERS.values()
    )
    return dist_km / ASSUMED_SPEED_KMH * 60 + FIXED_OVERHEAD_MIN, dist_km


def _towns_with_population():
    return [
        t for t in sorted(TOWN_COORDS.keys())
        if TOWN_POPULATION_2024.get(t) is not None
    ]


@lru_cache(maxsize=1)
def annualised_household_income() -> tuple:
    """Return (annual_household_income, source_year, raw_monthly_per_member).

    Applies the two dimensional corrections described in the module
    docstring: x12 to annualise, x AVG_HOUSEHOLD_SIZE to go from
    per-household-member to per-household.
    """
    _, median_income_df = load_income_data()
    hdb = load_hdb_data()
    latest_year = int(hdb["year_int"].max())

    rows = median_income_df[median_income_df["year"] == latest_year]
    if rows.empty:
        rows = median_income_df[median_income_df["year"] == median_income_df["year"].max()]
    source_year = int(rows["year"].iloc[0])
    monthly_per_member = float(rows["income"].iloc[0])

    annual_household = monthly_per_member * MONTHS_PER_YEAR * AVG_HOUSEHOLD_SIZE
    return annual_household, source_year, monthly_per_member


@lru_cache(maxsize=1)
def _real_town_features() -> pd.DataFrame:
    towns = _towns_with_population()

    hdb = load_hdb_data()
    latest_year = int(hdb["year_int"].max())
    annual_household_income, _, _ = annualised_household_income()

    avg_price_by_town = (
        hdb[hdb["year_int"] == latest_year].groupby("town")["median_resale_price"].mean()
    )

    # Rail: LRT included, assigned by nearest town centroid (see loader).
    rail = load_rail_stations()
    rail = rail[rail["operational"]]
    station_counts = rail.groupby("town").size()
    lrt_counts = rail[rail["mode"] == "LRT"].groupby("town").size()

    # Healthcare: distance to nearest hospital (real coords for all 28) plus
    # polyclinic count. NHG polyclinics carry town-centroid placeholder
    # coordinates, so they are counted, never distance-measured.
    hosp = load_hospitals_raw()
    hosp_pts = hosp.dropna(subset=["lat", "lng"])[["lat", "lng"]].to_numpy()
    poly = load_polyclinics_raw()
    poly_counts = poly.groupby("town").size()

    estate = load_estate_maturity_raw().set_index("town")

    rows = []
    for t in towns:
        lat, lon = TOWN_COORDS[t]
        pop = TOWN_POPULATION_2024[t]

        avg_price = avg_price_by_town.get(t, np.nan)
        price_to_income = (
            avg_price / annual_household_income if pd.notna(avg_price) else np.nan
        )

        station_count = int(station_counts.get(t, 0))
        polyclinic_count = int(poly_counts.get(t, 0))

        nearest_hosp_km = (
            min(_haversine(lat, lon, h[0], h[1]) for h in hosp_pts)
            if len(hosp_pts) else np.nan
        )

        estate_key = ESTATE_NAME_ALIASES.get(t, t)
        estate_age = (
            float(estate.loc[estate_key, "estimated_avg_estate_age_years_as_of_2026"])
            if estate_key in estate.index else np.nan
        )

        commute_min, job_dist_km = _commute_minutes(lat, lon)

        rows.append({
            "town": t, "lat": lat, "lng": lon, "population": pop,
            "avg_resale_price": avg_price,
            "price_to_income_ratio": price_to_income,
            "station_count": station_count,
            "lrt_count": int(lrt_counts.get(t, 0)),
            "station_count_per_10k": station_count / (pop / 10000),
            "polyclinic_count": polyclinic_count,
            "polyclinic_per_100k": polyclinic_count / (pop / 100000),
            "nearest_hospital_km": nearest_hosp_km,
            "estate_age_years": estate_age,
            "estate_age_imputed": not (estate_key in estate.index),
            "commute_minutes": commute_min,
            "commute_is_degenerate": job_dist_km < 0.5,
        })

    return pd.DataFrame(rows)


def _normalise(series: pd.Series, invert: bool = False) -> np.ndarray:
    """Min-max to [0,1], NaN-safe. NaNs are filled with the column median
    AFTER scaling so a town with one missing factor is not silently deleted
    from the whole analysis."""
    vals = series.to_numpy(dtype=float).reshape(-1, 1)
    mask = ~np.isnan(vals).flatten()
    out = np.full(len(vals), np.nan)
    if mask.sum() == 0:
        return np.zeros(len(vals))
    scaled = MinMaxScaler().fit_transform(vals[mask].reshape(-1, 1)).flatten()
    out[mask] = 1 - scaled if invert else scaled
    median_fill = np.nanmedian(out)
    return np.where(np.isnan(out), median_fill, out)


@lru_cache(maxsize=8)
def compute_equity_scores(weights_key: str = "default"):
    """Composite access-and-affordability score per town.

    `weights_key` selects a named weighting from WEIGHT_SCENARIOS. The old
    signature took an unused `year=None` that was silently ignored by
    callers; it has been removed rather than left as a trap.
    """
    weights = WEIGHT_SCENARIOS.get(weights_key, DEFAULT_WEIGHTS)
    df = _real_town_features().reset_index(drop=True)

    scored = pd.DataFrame({
        "affordability": _normalise(df["price_to_income_ratio"], invert=True),
        "transit_access": _normalise(df["station_count_per_10k"]),
        "healthcare_access": (
            _normalise(df["nearest_hospital_km"], invert=True) * 0.5
            + _normalise(df["polyclinic_per_100k"]) * 0.5
        ),
        "commute_access": _normalise(df["commute_minutes"], invert=True),
        "estate_modernity": _normalise(df["estate_age_years"], invert=True),
    })

    w = np.array([weights[c] for c in FACTOR_COLUMNS], dtype=float)
    w = w / w.sum()
    composite = (scored[FACTOR_COLUMNS].to_numpy() * w).sum(axis=1)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(scored[FACTOR_COLUMNS].to_numpy())

    out = df[["town", "lat", "lng"]].copy()
    for c in FACTOR_COLUMNS:
        out[c] = scored[c].round(3)
    out["composite_score"] = np.round(composite, 3)
    out["cluster"] = clusters.astype(int)
    out["estate_age_imputed"] = df["estate_age_imputed"]
    out["commute_is_degenerate"] = df["commute_is_degenerate"]
    return out


WEIGHT_SCENARIOS = {
    "default": DEFAULT_WEIGHTS,
    "affordability_led": {**DEFAULT_WEIGHTS, "affordability": 3.0},
    "transit_led": {**DEFAULT_WEIGHTS, "transit_access": 3.0},
    "healthcare_led": {**DEFAULT_WEIGHTS, "healthcare_access": 3.0},
    "drop_estate_age": {**DEFAULT_WEIGHTS, "estate_modernity": 0.0},
    "drop_commute": {**DEFAULT_WEIGHTS, "commute_access": 0.0},
}


def compute_weight_sensitivity() -> pd.DataFrame:
    """How much does the town ranking depend on the equal-weighting choice?

    Returns Spearman rank correlation of each alternative weighting against
    the default, plus the biggest rank mover. Exists so the equal weighting
    is testable rather than asserted.
    """
    base = compute_equity_scores("default").set_index("town")["composite_score"]
    base_rank = base.rank(ascending=False)

    rows = []
    for name in WEIGHT_SCENARIOS:
        if name == "default":
            continue
        alt = compute_equity_scores(name).set_index("town")["composite_score"]
        alt_rank = alt.rank(ascending=False)
        rho = base_rank.corr(alt_rank, method="spearman")
        delta = (alt_rank - base_rank).abs()
        rows.append({
            "scenario": name,
            "spearman_vs_default": round(rho, 3),
            "max_rank_shift": int(delta.max()),
            "biggest_mover": delta.idxmax(),
        })
    return pd.DataFrame(rows)


def compute_factor_details(town: str):
    """Raw (non-normalised) factor values for one town."""
    df = _real_town_features()
    row = df[df["town"] == town]
    if row.empty:
        return None
    row = row.iloc[0]
    annual_income, income_year, monthly_per_member = annualised_household_income()
    return {
        "price_to_annual_household_income": (
            round(row["price_to_income_ratio"], 2)
            if pd.notna(row["price_to_income_ratio"]) else None
        ),
        "avg_resale_price": (
            round(row["avg_resale_price"]) if pd.notna(row["avg_resale_price"]) else None
        ),
        "assumed_annual_household_income": round(annual_income),
        "income_source_year": income_year,
        "income_raw_monthly_per_member": monthly_per_member,
        "rail_stations": int(row["station_count"]),
        "of_which_lrt": int(row["lrt_count"]),
        "polyclinics": int(row["polyclinic_count"]),
        "nearest_hospital_km": (
            round(row["nearest_hospital_km"], 2)
            if pd.notna(row["nearest_hospital_km"]) else None
        ),
        "estate_age_years": (
            round(row["estate_age_years"], 1)
            if pd.notna(row["estate_age_years"]) else None
        ),
        "estate_age_imputed": bool(row["estate_age_imputed"]),
        "commute_minutes": round(row["commute_minutes"], 1),
        "commute_is_degenerate": bool(row["commute_is_degenerate"]),
        "population": int(row["population"]),
    }


# ---------------------------------------------------------------------------
# Getis-Ord Gi* hotspot analysis on the composite score.
# ---------------------------------------------------------------------------

def compute_getis_ord(
    k_neighbors: int = 5,
    weights_key: str = "default",
    n_permutations: int = 9999,
    random_state: int = 42,
) -> pd.DataFrame:
    """Getis-Ord Gi* z-scores using a k-nearest-neighbour binary weights
    matrix, with CONDITIONAL PERMUTATION INFERENCE for significance.

    Implemented directly because esda/libpysal are unavailable in the offline
    build environment -- but the absence of those libraries is not a reason to
    fall back on normal-approximation p-values, which are a poor guide at
    n=26. Instead, for each town i we hold x_i fixed, randomly reshuffle the
    remaining n-1 values among the other towns `n_permutations` times,
    recompute Gi*_i under each reshuffle, and report the share of permuted
    statistics at least as extreme as the observed one. That empirical
    (pseudo) p-value makes no distributional assumption.

    Two-sided pseudo p, following the standard (r + 1) / (permutations + 1)
    convention so p is never reported as exactly zero. The finest resolvable
    p-value is therefore 1 / (n_permutations + 1); with the default 9999
    permutations that is 1e-4.

    REMAINING CAVEAT: pseudo p-values are not corrected for multiple
    comparisons across the 26 towns. `p_sim_fdr` applies a
    Benjamini-Hochberg correction and `significant_fdr` uses it at the 5%
    level -- prefer those columns when making a claim about any specific town.
    """
    df = compute_equity_scores(weights_key).reset_index(drop=True)
    n = len(df)
    coords = df[["lat", "lng"]].to_numpy()
    x = df["composite_score"].to_numpy()

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = _haversine(*coords[i], *coords[j])

    W = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(dist[i])[:k_neighbors]:
            W[i, j] = 1
        W[i, i] = 1  # Gi* includes the focal unit

    x_bar, s = x.mean(), x.std(ddof=0)

    def _gi_for(i, values):
        """Gi*_i given a full value vector (weights are fixed by geography)."""
        w_i = W[i]
        sum_w, sum_wx = w_i.sum(), (w_i * values).sum()
        denom = s * math.sqrt((n * (w_i ** 2).sum() - sum_w ** 2) / (n - 1))
        return (sum_wx - x_bar * sum_w) / denom if denom > 0 else 0.0

    gi = np.array([_gi_for(i, x) for i in range(n)])

    # --- Conditional permutation inference -------------------------------
    # For town i: fix x_i, shuffle the other n-1 values, recompute Gi*_i.
    rng = np.random.default_rng(random_state)
    p_sim = np.zeros(n)
    for i in range(n):
        others = np.delete(x, i)
        permuted_gi = np.empty(n_permutations)
        scratch = np.empty(n)
        scratch[i] = x[i]
        idx = np.arange(n) != i
        for b in range(n_permutations):
            scratch[idx] = rng.permutation(others)
            permuted_gi[b] = _gi_for(i, scratch)
        # Two-sided: how often is |permuted| >= |observed|
        extreme = np.count_nonzero(np.abs(permuted_gi) >= abs(gi[i]))
        p_sim[i] = (extreme + 1) / (n_permutations + 1)

    # --- Benjamini-Hochberg FDR correction across the n towns -------------
    order = np.argsort(p_sim)
    ranked = p_sim[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]  # enforce monotonicity
    p_fdr = np.empty(n)
    p_fdr[order] = np.minimum(bh, 1.0)

    out = df[["town", "composite_score"]].copy()
    out["gi_star_z"] = gi.round(3)
    out["p_sim"] = p_sim.round(4)
    out["p_sim_fdr"] = p_fdr.round(4)
    out["significant_fdr"] = p_fdr < 0.05
    out["cluster_label"] = np.where(
        ~out["significant_fdr"], "No clustering",
        np.where(out["gi_star_z"] > 0, "High cluster (FDR<0.05)", "Low cluster (FDR<0.05)"),
    )
    return out.sort_values("gi_star_z", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shapley attribution -- now against the 982k-row transaction model.
# ---------------------------------------------------------------------------

SHAPLEY_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "processed_data", "transaction_shapley_model.joblib",
)


@lru_cache(maxsize=1)
def _load_shapley_model():
    """Load the compact interpretable model trained on all 982,011
    transactions by pipeline/train_transaction_model.py.

    This replaced a RandomForestRegressor fit on 23 town rows. The Shapley
    math there was correct but the object being explained was noise; a
    depth-4 forest on 23 samples is not a model of anything. Six features are
    used so exact (2^6 = 64 coalition) Shapley remains tractable without
    approximation.
    """
    if not os.path.exists(SHAPLEY_MODEL_PATH):
        return None
    import joblib
    return joblib.load(SHAPLEY_MODEL_PATH)


def compute_shapley_price_drivers(town: str) -> dict:
    """Exact Shapley attribution for a representative flat in `town`, against
    the transaction-level model. Returns None if that model has not been
    trained yet (run pipeline/train_transaction_model.py)."""
    bundle = _load_shapley_model()
    if bundle is None:
        return None

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    background = np.asarray(bundle["background"], dtype=float)
    town_profiles = bundle["town_profiles"]

    if town not in town_profiles:
        return None
    x_town = np.asarray(town_profiles[town], dtype=float)

    n_feat = len(feature_names)

    def predict(present: set) -> float:
        x = background.copy()
        for i in present:
            x[i] = x_town[i]
        return float(model.predict(x.reshape(1, -1))[0])

    baseline = predict(set())
    idx = list(range(n_feat))
    shapley = np.zeros(n_feat)
    for i in idx:
        others = [j for j in idx if j != i]
        total = 0.0
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s = set(subset)
                weight = (
                    math.factorial(len(s)) * math.factorial(n_feat - len(s) - 1)
                ) / math.factorial(n_feat)
                total += weight * (predict(s | {i}) - predict(s))
        shapley[i] = total

    return {
        "town": town,
        "baseline_price": round(baseline),
        "predicted_price": round(predict(set(idx))),
        "n_training_rows": bundle.get("n_training_rows"),
        "contributions": {
            feature_names[i]: round(shapley[i]) for i in idx
        },
    }
