"""Data loading and caching for the Singapore HDB Access & Affordability Analyser dashboard."""

import os
import csv
import json
import re
import math
from functools import lru_cache

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "processed_data")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "raw_data")

# Approximate town centroid coordinates (lat, lng) for Singapore HDB towns
TOWN_COORDS = {
    "Ang Mo Kio": (1.3691, 103.8456),
    "Bedok": (1.3236, 103.9273),
    "Bishan": (1.3508, 103.8495),
    "Bukit Batok": (1.3488, 103.7494),
    "Bukit Merah": (1.2805, 103.8238),
    "Bukit Panjang": (1.3786, 103.7668),
    "Bukit Timah": (1.3294, 103.8021),
    "Central Area": (1.2850, 103.8530),
    "Choa Chu Kang": (1.3770, 103.7450),
    "Clementi": (1.3150, 103.7650),
    "Geylang": (1.3200, 103.8900),
    "Hougang": (1.3615, 103.8867),
    "Jurong East": (1.3329, 103.7414),
    "Jurong West": (1.3400, 103.7100),
    "Kallang/Whampoa": (1.3150, 103.8580),
    "Lim Chu Kang": (1.4300, 103.7200),
    "Marine Parade": (1.3020, 103.9050),
    "Pasir Ris": (1.3720, 103.9490),
    "Punggol": (1.4020, 103.9100),
    "Queenstown": (1.2940, 103.8000),
    "Sembawang": (1.4480, 103.8180),
    "Sengkang": (1.3920, 103.8950),
    "Serangoon": (1.3550, 103.8740),
    "Tampines": (1.3490, 103.9570),
    "Toa Payoh": (1.3330, 103.8470),
    "Woodlands": (1.4360, 103.7890),
    "Yishun": (1.4300, 103.8350),
}

# MRT station coordinates (major stations by line)
MRT_STATIONS = {
    "Ang Mo Kio": (1.3693, 103.8492),
    "Bishan": (1.3506, 103.8484),
    "Braddell": (1.3404, 103.8468),
    "Toa Payoh": (1.3328, 103.8475),
    "Novena": (1.3204, 103.8438),
    "Newton": (1.3114, 103.8383),
    "Orchard": (1.3040, 103.8318),
    "Somerset": (1.3004, 103.8387),
    "Dhoby Ghaut": (1.2992, 103.8457),
    "City Hall": (1.2931, 103.8527),
    "Raffles Place": (1.2837, 103.8510),
    "Marina Bay": (1.2768, 103.8548),
    "Marina South Pier": (1.2712, 103.8634),
    "Harbourfront": (1.2648, 103.8220),
    "Chinatown": (1.2846, 103.8430),
    "Clarke Quay": (1.2894, 103.8460),
    "Little India": (1.3067, 103.8490),
    "Farrer Park": (1.3122, 103.8535),
    "Boon Keng": (1.3198, 103.8616),
    "Potong Pasir": (1.3317, 103.8690),
    "Woodleigh": (1.3396, 103.8711),
    "Serangoon": (1.3506, 103.8734),
    "Kovan": (1.3614, 103.8857),
    "Hougang": (1.3714, 103.8931),
    "Buangkok": (1.3825, 103.8932),
    "Sengkang": (1.3914, 103.8951),
    "Punggol": (1.4052, 103.9020),
    "Jurong East": (1.3329, 103.7426),
    "Chinese Garden": (1.3424, 103.7327),
    "Lakeside": (1.3444, 103.7213),
    "Boon Lay": (1.3385, 103.7061),
    "Pioneer": (1.3377, 103.6973),
    "Joo Koon": (1.3271, 103.6786),
    "Gul Circle": (1.3195, 103.6608),
    "Tuas Crescent": (1.3206, 103.6482),
    "Tuas West Road": (1.3209, 103.6377),
    "Tuas Link": (1.3215, 103.6265),
    "Expo": (1.3347, 103.9601),
    "Changi Airport": (1.3575, 103.9886),
    "Tanah Merah": (1.3274, 103.9466),
    "Simei": (1.3433, 103.9537),
    "Tampines": (1.3523, 103.9448),
    "Pasir Ris": (1.3728, 103.9493),
    "Khatib": (1.4175, 103.8277),
    "Yishun": (1.4295, 103.8351),
    "Sembawang": (1.4491, 103.8198),
    "Admiralty": (1.4406, 103.8013),
    "Woodlands": (1.4370, 103.7864),
    "Marsiling": (1.4326, 103.7748),
    "Kranji": (1.4255, 103.7534),
    "Bukit Batok": (1.3490, 103.7497),
    "Bukit Gombak": (1.3588, 103.7521),
    "Choa Chu Kang": (1.3845, 103.7448),
    "Yew Tee": (1.3975, 103.7474),
    "Sungei Kadut": (1.4156, 103.7636),
    "Bukit Panjang": (1.3786, 103.7647),
    "Fajar": (1.3828, 103.7706),
    "Segar": (1.3867, 103.7772),
    "Hillview": (1.3627, 103.7641),
    "Cashew": (1.3757, 103.7648),
    "Beauty World": (1.3415, 103.7776),
    "King Albert Park": (1.3358, 103.7839),
    "Sixth Avenue": (1.3308, 103.7951),
    "Tan Kah Kee": (1.3254, 103.8078),
    "Botanic Gardens": (1.3222, 103.8147),
    "Stevens": (1.3199, 103.8256),
    "Mountbatten": (1.3069, 103.8823),
    "Dakota": (1.3084, 103.8877),
    "Paya Lebar": (1.3181, 103.8922),
    "MacPherson": (1.3267, 103.8892),
    "Tai Seng": (1.3358, 103.8875),
    "Bartley": (1.3425, 103.8788),
    "Lorong Chuan": (1.3517, 103.8639),
    "Marymount": (1.3492, 103.8395),
    "Caldecott": (1.3381, 103.8392),
    "Farrer Road": (1.3171, 103.8079),
    "Holland Village": (1.3116, 103.7955),
    "Buona Vista": (1.3066, 103.7910),
    "One-North": (1.2994, 103.7869),
    "Kent Ridge": (1.2933, 103.7845),
    "Haw Par Villa": (1.2833, 103.7880),
    "Pasir Panjang": (1.2771, 103.7890),
    "Labrador Park": (1.2721, 103.8026),
    "Telok Blangah": (1.2704, 103.8100),
}

# Hospital coordinates (major hospitals in Singapore)
HOSPITALS = {
    "Singapore General Hospital": (1.2785, 103.8340),
    "National University Hospital": (1.2942, 103.7828),
    "Tan Tock Seng Hospital": (1.3205, 103.8450),
    "Khoo Teck Puat Hospital": (1.4190, 103.8375),
    "Changi General Hospital": (1.3380, 103.9580),
    "KK Women\'s and Children\'s Hospital": (1.3040, 103.8425),
    "Mount Elizabeth Hospital": (1.3048, 103.8335),
    "Gleneagles Hospital": (1.3098, 103.8275),
    "Raffles Hospital": (1.3000, 103.8550),
    "Ng Teng Fong General Hospital": (1.3329, 103.7417),
    "Sengkang General Hospital": (1.3900, 103.8950),
    "Woodlands Health Campus": (1.4360, 103.7890),
}


@lru_cache(maxsize=1)
def load_hdb_data() -> pd.DataFrame:
    """Load aggregated HDB town-quarter data."""
    path = os.path.join(DATA_DIR, "hdb-town-quarter.csv")
    df = pd.read_csv(path)

    # Add town coordinates
    coords_df = pd.DataFrame.from_dict(
        TOWN_COORDS, orient="index", columns=["lat", "lng"]
    )
    coords_df.index.name = "town"
    df = df.merge(coords_df.reset_index(), on="town", how="left")

    # Parse quarter into sortable numeric
    df["quarter_sort"] = df["quarter"].apply(_quarter_to_num)
    df["year_int"] = df["year"].astype(int)
    return df


@lru_cache(maxsize=1)
def load_income_data() -> pd.DataFrame:
    """Load national-level household income percentiles."""
    path = os.path.join(
        RAW_DIR,
        "MonthlyHouseholdEmploymentIncomePerHouseholdMemberExcludingEmployer"
        "CPFContributionsAmongResidentEmployedHouseholdsatSelectedPercentiles"
        "HouseholdIncomeAnnual20002025.csv",
    )
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            percentile = row["Dollar"]
            for year, val in row.items():
                if year == "Dollar":
                    continue
                try:
                    rows.append({
                        "year": int(year),
                        "percentile": percentile,
                        "income": float(val),
                    })
                except ValueError:
                    pass
    df = pd.DataFrame(rows)
    # Extract median row
    median_df = df[df["percentile"].str.contains("Median", case=False)].copy()
    if median_df.empty:
        median_df = df[df["percentile"] == "50th"].copy()
    return df, median_df


def _quarter_to_num(q: str) -> float:
    """Convert '1990-Q1' to 1990.0 etc."""
    try:
        parts = q.split("-Q")
        return int(parts[0]) + (int(parts[1]) - 1) / 4.0
    except (IndexError, ValueError):
        return 0.0


@lru_cache(maxsize=1)
def get_town_list() -> list[str]:
    """Get sorted list of towns with data."""
    df = load_hdb_data()
    return sorted(df["town"].unique())


@lru_cache(maxsize=1)
def get_flat_types() -> list[str]:
    """Get sorted list of flat types."""
    df = load_hdb_data()
    return sorted(df["flat_type"].unique())


@lru_cache(maxsize=1)
def get_year_range() -> tuple[int, int]:
    """Get min and max year."""
    df = load_hdb_data()
    return int(df["year_int"].min()), int(df["year_int"].max())


def get_mrt_stations_df() -> pd.DataFrame:
    """Return MRT station coordinates as DataFrame."""
    return pd.DataFrame.from_dict(
        MRT_STATIONS, orient="index", columns=["lat", "lng"]
    ).reset_index().rename(columns={"index": "station"})


def get_hospitals_df() -> pd.DataFrame:
    """Return hospital coordinates as DataFrame."""
    return pd.DataFrame.from_dict(
        HOSPITALS, orient="index", columns=["lat", "lng"]
    ).reset_index().rename(columns={"index": "hospital"})


# ---------------------------------------------------------------------------
# Real raw-data loaders (used by equity_factors.py). These read the actual
# CSVs shipped in raw_data/ instead of re-typing numbers by hand.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _mrt_exit_centroids() -> dict:
    """Average LTA's real per-exit coordinates (raw_data/lta-mrt-station-exits.geojson,
    source: data.gov.sg 'LTA MRT Station Exit (GEOJSON)', Aug 2025 snapshot)
    into one centroid per station. Returns {name: (lat, lng, mode)}.

    Mode is read off the station's own name suffix in the source data
    ("... MRT STATION" vs "... LRT STATION"), not inferred.
    """
    path = os.path.join(RAW_DIR, "lta-mrt-station-exits.geojson")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sums = {}
    for feat in data["features"]:
        raw_name = feat["properties"]["STATION_NA"]
        mode = "LRT" if re.search(r"\bLRT\b", raw_name, flags=re.I) else "MRT"
        name = re.sub(r"\s+(MRT|LRT) STATION\s*$", "", raw_name, flags=re.I).strip()
        lon, lat = feat["geometry"]["coordinates"]
        s = sums.setdefault(name, [0.0, 0.0, 0, mode])
        s[0] += lat
        s[1] += lon
        s[2] += 1

    return {
        name: (lat_sum / n, lon_sum / n, mode)
        for name, (lat_sum, lon_sum, n, mode) in sums.items()
    }


@lru_cache(maxsize=1)
def load_rail_stations() -> pd.DataFrame:
    """Authoritative rail-station table: EVERY station in LTA's exit dataset,
    heavy rail (MRT) and light rail (LRT) alike.

    WHY THIS REPLACED THE OLD mrt-stations.csv-DRIVEN LOADER
    ---------------------------------------------------------
    raw_data/mrt-stations.csv contains 115 operational stations and, despite
    the column naming, contains ZERO LRT stations (verified: no row matches
    /LRT/). Driving the transit factor off that file silently erased the
    Bukit Panjang, Sengkang and Punggol LRT loops -- roughly 41 stations --
    which are precisely the feeder networks that make those towns
    transit-accessible. Punggol and Sengkang were consequently scored at 1-2
    stations each, versus 13 for Central Area, and that bias propagated into
    the composite score, the k-means clusters and the Gi* hotspot map.

    The LTA exit GEOJSON does include LRT (41 of 187 stations), so this
    loader treats the GEOJSON as the source of truth and uses
    mrt-stations.csv only to cross-check operational status where the names
    match. Stations present in the GEOJSON but absent from that CSV are
    retained and flagged via `in_status_csv=False` rather than dropped,
    since the CSV is demonstrably incomplete.
    """
    centroids = _mrt_exit_centroids()

    status_path = os.path.join(RAW_DIR, "mrt-stations.csv")
    status_df = pd.read_csv(status_path)
    operational = {
        n.strip().lower()
        for n in status_df.loc[status_df["status"] == "operational", "station_name"]
    }
    known = {n.strip().lower() for n in status_df["station_name"]}

    rows = []
    for name, (lat, lng, mode) in centroids.items():
        key = name.strip().lower()
        rows.append({
            "station_name": name,
            "mode": mode,
            "lat": lat,
            "lng": lng,
            "in_status_csv": key in known,
            # Present in the exit dataset => physically built with exits.
            # Treat as operational unless the status CSV explicitly says otherwise.
            "operational": (key in operational) or (key not in known),
        })

    df = pd.DataFrame(rows)
    df["town"] = df.apply(lambda r: _nearest_town(r["lat"], r["lng"]), axis=1)
    return df


@lru_cache(maxsize=1)
def load_mrt_stations_raw() -> pd.DataFrame:
    """Backwards-compatible alias for load_rail_stations() with the old column
    names, so existing callers (e.g. map_view) keep working.
    """
    df = load_rail_stations().copy()
    df["geocoded"] = True
    return df


# Manually curated mapping from the free-text "area" field in hospitals.csv to
# the nearest HDB town -- RETAINED ONLY AS A FALLBACK now that real
# coordinates exist for every hospital (see load_hospitals_raw). Kept here in
# case a future hospital is added to hospitals.csv without a matching
# coordinate row.
HOSPITAL_AREA_TO_TOWN = {
    "Simei": "Tampines", "Rochor": "Kallang/Whampoa", "Sengkang": "Sengkang",
    "Outram": "Bukit Merah", "Yishun": "Yishun", "Novena": "Toa Payoh",
    "Woodlands": "Woodlands", "Queenstown": "Queenstown", "Kent Ridge": "Clementi",
    "Jurong East": "Jurong East", "Marymount": "Bishan", "Bukit Timah": "Bukit Timah",
    "Farrer Park": "Kallang/Whampoa", "Tanglin": "Bukit Timah", "Orchard": "Central Area",
    "Joo Chiat": "Geylang", "Bugis": "Central Area", "Thomson": "Bishan",
    "Bukit Merah": "Bukit Merah", "Bedok": "Bedok", "Tengah": "Choa Chu Kang",
    "Ang Mo Kio": "Ang Mo Kio", "Bukit Batok": "Bukit Batok", "Hougang": "Hougang",
}


@lru_cache(maxsize=1)
def load_hospitals_raw() -> pd.DataFrame:
    """Load the real hospital inventory from raw_data/hospitals.csv, geocoded
    with real coordinates (raw_data/hospital-coordinates.csv, user-supplied,
    verified to match all 28 operational hospital names exactly) and assigned
    to towns by nearest centroid. Falls back to HOSPITAL_AREA_TO_TOWN only if
    a hospital is missing a coordinate row. 'upcoming' facilities (not yet
    built) are excluded from present-day access counts.
    """
    path = os.path.join(RAW_DIR, "hospitals.csv")
    df = pd.read_csv(path)
    df = df[df["status"] == "operational"].copy()

    coords_path = os.path.join(RAW_DIR, "hospital-coordinates.csv")
    coords = pd.read_csv(coords_path).set_index("hospital_name")
    df["lat"] = df["hospital_name"].map(lambda n: coords["lat"].get(n))
    df["lng"] = df["hospital_name"].map(lambda n: coords["lng"].get(n))

    has_coords = df["lat"].notna()
    df.loc[has_coords, "town"] = df.loc[has_coords].apply(
        lambda r: _nearest_town(r["lat"], r["lng"]), axis=1
    )
    df.loc[~has_coords, "town"] = df.loc[~has_coords, "area"].map(HOSPITAL_AREA_TO_TOWN)
    df = df.dropna(subset=["town"])
    return df


def _nearest_town(lat, lon):
    best_town, best_dist = None, float("inf")
    for t, (tlat, tlon) in TOWN_COORDS.items():
        d = math.dist([lat, lon], [tlat, tlon])  # planar approx fine at this scale
        if d < best_dist:
            best_town, best_dist = t, d
    return best_town


# Manual corrections for cases where nearest-centroid assignment is wrong
# because a location sits near a town border (verified case: "Tampines
# North" is administratively part of Tampines, but its coordinates are
# closer to the Pasir Ris town centroid than the Tampines centroid).
POLYCLINIC_TOWN_OVERRIDES = {
    "SHP-Tampines North": "Tampines",
}


@lru_cache(maxsize=1)
def load_polyclinics_raw() -> pd.DataFrame:
    """Load real polyclinic locations, covering all 3 public healthcare
    clusters (28 total -- the full Singapore public polyclinic network):

    - raw_data/polyclinics.csv        SingHealth (10), real coordinates
    - raw_data/polyclinics_nuhs.csv   NUHS (8), real coordinates
    - raw_data/polyclinics_nhg.csv    NHG (10), NO coordinates supplied --
      each facility is named directly after its home town (e.g. "Hougang
      Polyclinic" -> Hougang), so town is assigned by that name match rather
      than geocoded. lat/lng for these rows are set to the town centroid
      purely for map display, not an independently verified location.
      Two non-literal names are resolved by hand in polyclinics_nhg.csv:
      Khatib -> Yishun (Khatib is a subzone within Yishun town) and
      Kallang -> Kallang/Whampoa (matching this project's HDB town naming).

    Coordinate-based rows are assigned a town by nearest centroid, with
    POLYCLINIC_TOWN_OVERRIDES applied for verified near-border cases.
    """
    sh = pd.read_csv(os.path.join(RAW_DIR, "polyclinics.csv"))
    sh["operator"] = "SingHealth"
    sh = sh[["polyclinic", "lat", "lng", "operator"]]

    nuhs = pd.read_csv(os.path.join(RAW_DIR, "polyclinics_nuhs.csv"))

    geocoded = pd.concat([sh, nuhs], ignore_index=True)
    geocoded["town"] = geocoded.apply(lambda r: _nearest_town(r["lat"], r["lng"]), axis=1)
    geocoded["town"] = geocoded.apply(
        lambda r: POLYCLINIC_TOWN_OVERRIDES.get(r["polyclinic"], r["town"]), axis=1
    )
    geocoded["town_geocoded"] = True

    nhg = pd.read_csv(os.path.join(RAW_DIR, "polyclinics_nhg.csv"))
    nhg["lat"] = nhg["town"].map(lambda t: TOWN_COORDS.get(t, (None, None))[0])
    nhg["lng"] = nhg["town"].map(lambda t: TOWN_COORDS.get(t, (None, None))[1])
    nhg["town_geocoded"] = False
    nhg = nhg[["polyclinic", "lat", "lng", "operator", "town", "town_geocoded"]]

    return pd.concat([geocoded, nhg], ignore_index=True)


@lru_cache(maxsize=1)
def load_estate_maturity_raw() -> pd.DataFrame:
    """Load real estate-age data from raw_data/estate-maturity.csv.

    NOTE: at least one row in the source file (Sengkang) has an unescaped
    comma inside the unquoted 'notes' field ("Includes Compassvale, Rivervale,
    Anchorvale, Fernvale"), which breaks pandas' default C parser. Rather than
    silently dropping that row, extra comma-split fragments are rejoined back
    into 'notes' here.
    """
    path = os.path.join(RAW_DIR, "estate-maturity.csv")
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        rows = []
        for row in reader:
            if len(row) > n_cols:
                row = row[: n_cols - 1] + [",".join(row[n_cols - 1:])]
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    numeric_cols = [
        "estimated_avg_estate_age_years_as_of_2026",
        "number_of_blocks_approx", "total_units_approx",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df