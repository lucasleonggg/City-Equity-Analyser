"""Merge all 6 HDB resale CSV files into a single dataset with normalized schema."""

import csv
import os
import re

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hdb-all.csv")
FILES = [
    "hdb-1990-1999.csv",
    "hdb-2000-2012.csv",
    "hdb-2012-2014.csv",
    "hdb-2015-2016.csv",
    "hdb-2017-onwards.csv",
]

# Target columns (union of all schemas, in consistent order)
TARGET_COLS = [
    "month", "town", "flat_type", "block", "street_name",
    "storey_range", "floor_area_sqm", "flat_model",
    "lease_commence_date", "remaining_lease", "resale_price", "source_file",
]


def parse_remaining_lease_text(text: str) -> float | None:
    """Convert '61 years 04 months' to years (float). Handles sing/plural."""
    if not text or text.strip() == "":
        return None
    text = text.strip().lower()
    m = re.match(r"(\d+)\s*years?\s*(?:(\d+)\s*months?)?", text)
    if m:
        years = float(m.group(1))
        months = float(m.group(2)) if m.group(2) else 0.0
        return years + months / 12.0
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_with_encoding(filepath: str):
    """Read a CSV using comma delimiter."""
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames, rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_rows = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=TARGET_COLS, extrasaction="ignore")
        writer.writeheader()

        for fname in FILES:
            filepath = os.path.join(RAW_DIR, fname)
            if not os.path.exists(filepath):
                print(f"WARNING: {fname} not found, skipping")
                continue

            fieldnames, rows = read_csv_with_encoding(filepath)
            print(f"{fname}: {len(rows)} rows, cols={fieldnames}")

            has_id = "_id" in fieldnames
            has_remaining = "remaining_lease" in fieldnames

            for row in rows:
                # Build normalized row
                out_row = {col: row.get(col, "") for col in TARGET_COLS}
                out_row["source_file"] = fname

                # Drop _id (just don't copy it — extrasaction='ignore' handles it)

                # Normalize remaining_lease
                if has_remaining:
                    rl = row.get("remaining_lease", "").strip()
                    if "year" in rl.lower() or "month" in rl.lower():
                        rl = parse_remaining_lease_text(rl)
                        out_row["remaining_lease"] = str(rl) if rl is not None else ""
                    else:
                        out_row["remaining_lease"] = rl
                else:
                    out_row["remaining_lease"] = ""

                # Ensure numeric fields are clean
                for num_field in ["floor_area_sqm", "resale_price"]:
                    val = out_row[num_field].strip()
                    try:
                        float(val)
                    except ValueError:
                        out_row[num_field] = ""

                writer.writerow(out_row)
                total_rows += 1

    print(f"\nMerged {total_rows} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()