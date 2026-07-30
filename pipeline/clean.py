"""Clean and standardize the merged HDB dataset."""

import csv
import os
from datetime import datetime

INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "hdb-all.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "hdb-all-cleaned.csv")

# Standard flat_type mappings
FLAT_TYPE_MAP = {
    "1 room": "1-Room",
    "1-room": "1-Room",
    "2 room": "2-Room",
    "2-room": "2-Room",
    "3 room": "3-Room",
    "3-room": "3-Room",
    "4 room": "4-Room",
    "4-room": "4-Room",
    "5 room": "5-Room",
    "5-room": "5-Room",
    "executive": "Executive",
    "multi-generation": "Multi-Generation",
    "multigeneration": "Multi-Generation",
    "multi generation": "Multi-Generation",
    "terrace": "Terrace",
    "improved": "Improved",
    "new generation": "New Generation",
    "simplified": "Simplified",
    "model a": "Model A",
    "model a2": "Model A2",
    "standard": "Standard",
    "premium apartment": "Premium Apartment",
    "premium maisonette": "Premium Maisonette",
    "apartment": "Apartment",
    "maisonette": "Maisonette",
    "dbss": "DBSS",
    "type s1": "Type S1",
    "type s2": "Type S2",
    "adjoined flat": "Adjoined Flat",
    "improvement": "Improved",
}

# Town name standardization
TOWN_MAP = {
    "ang mo kio": "Ang Mo Kio",
    "bedok": "Bedok",
    "bishan": "Bishan",
    "bukit batok": "Bukit Batok",
    "bukit merah": "Bukit Merah",
    "bukit panjang": "Bukit Panjang",
    "bukit timah": "Bukit Timah",
    "central area": "Central Area",
    "central": "Central Area",
    "choa chu kang": "Choa Chu Kang",
    "clementi": "Clementi",
    "geylang": "Geylang",
    "hougang": "Hougang",
    "jurong east": "Jurong East",
    "jurong west": "Jurong West",
    "kallang/whampoa": "Kallang/Whampoa",
    "kallang whampoa": "Kallang/Whampoa",
    "pasir ris": "Pasir Ris",
    "punggol": "Punggol",
    "queenstown": "Queenstown",
    "sembawang": "Sembawang",
    "sengkang": "Sengkang",
    "serangoon": "Serangoon",
    "tampines": "Tampines",
    "toa payoh": "Toa Payoh",
    "woodlands": "Woodlands",
    "yishun": "Yishun",
    "lim chu kang": "Lim Chu Kang",
    "marine parade": "Marine Parade",
    "novena": "Novena",
    "outram": "Outram",
    "river valley": "River Valley",
    "rochor": "Rochor",
    "singapore": "Singapore",
    "southern islands": "Southern Islands",
    "stirling": "Stirling",
    "tanglin": "Tanglin",
    "tengah": "Tengah",
    "western islands": "Western Islands",
    "western water catchment": "Western Water Catchment",
}


def parse_month(month_str: str) -> datetime | None:
    """Parse month field. Format: 'YYYY-MM'."""
    month_str = month_str.strip()
    try:
        return datetime.strptime(month_str, "%Y-%m")
    except ValueError:
        pass
    try:
        return datetime.strptime(month_str, "%Y-%M")
    except ValueError:
        pass
    return None


def standardize_flat_type(ft: str) -> str:
    """Standardize flat_type to title-case."""
    ft = ft.strip().lower()
    return FLAT_TYPE_MAP.get(ft, ft.title())


def standardize_town(town: str) -> str:
    """Standardize town name."""
    t = town.strip().lower()
    return TOWN_MAP.get(t, town.strip().title())


def approx_remaining_lease(row: dict) -> str | None:
    est = compute_lease_years(row)
    return str(round(est, 2)) if est is not None else None


def compute_lease_years(row: dict) -> float | None:
    try:
        lcd = int(row["lease_commence_date"])
        month_d = parse_month(row["month"])
        if month_d is None:
            return None
        # HDB leases are 99 years from lease_commence_date
        lease_end = datetime(lcd + 99, 1, 1)
        remaining = (lease_end - month_d).days / 365.25
        return max(0.0, remaining)
    except (ValueError, TypeError):
        return None


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Cleaning {len(rows)} rows...")

    cleaned = []
    skipped = 0
    for row in rows:
        # Parse month
        month_dt = parse_month(row["month"])
        if month_dt is None:
            skipped += 1
            continue

        # Standardize categoricals
        row["town"] = standardize_town(row["town"])
        row["flat_type"] = standardize_flat_type(row["flat_type"])
        row["flat_model"] = row.get("flat_model", "").strip().title()

        # Ensure numeric types
        for field in ["floor_area_sqm", "resale_price"]:
            val = row.get(field, "").strip()
            try:
                row[field] = str(float(val))
            except (ValueError, TypeError):
                row[field] = ""

        # Parse remaining_lease: if empty, estimate from lease_commence_date
        rl = row.get("remaining_lease", "").strip()
        if not rl:
            est = approx_remaining_lease(row)
            if est:
                row["remaining_lease"] = est
            else:
                row["remaining_lease"] = ""
        else:
            try:
                row["remaining_lease"] = str(round(float(rl), 2))
            except ValueError:
                row["remaining_lease"] = ""

        # Extract year and quarter for downstream use
        row["year"] = str(month_dt.year)
        row["quarter"] = f"{month_dt.year}-Q{(month_dt.month - 1) // 3 + 1}"

        cleaned.append(row)

    # Write output.
    # NOTE: `year` and `quarter` are set on each row above, so by this point
    # they are ALREADY present in rows[0].keys(). Appending them again here
    # was producing a duplicated header (...,year,quarter,year,quarter) and
    # duplicate columns in hdb-all-cleaned.csv. Build the fieldname list from
    # the row keys and only append what is genuinely missing.
    fieldnames = list(cleaned[0].keys()) if cleaned else list(rows[0].keys())
    for extra in ("year", "quarter"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Cleaned: {len(cleaned)} rows written, {skipped} skipped (bad dates)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()