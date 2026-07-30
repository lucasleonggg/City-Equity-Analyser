"""Aggregate cleaned HDB data to town-quarter level for dashboard and modeling."""

import csv
import os
from collections import defaultdict

INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "hdb-all-cleaned.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "hdb-town-quarter.csv")

OUTPUT_COLS = [
    "town", "quarter", "year", "flat_type",
    "median_resale_price", "mean_resale_price",
    "median_floor_area_sqm", "mean_floor_area_sqm",
    "transaction_count",
    "median_remaining_lease_years", "mean_remaining_lease_years",
]


def median(values: list[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def main():
    groups: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"prices": [], "areas": [], "leases": [], "year": ""}
    )

    with open(INPUT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["town"], row["quarter"], row["flat_type"])

            price = row.get("resale_price", "").strip()
            area = row.get("floor_area_sqm", "").strip()
            lease = row.get("remaining_lease", "").strip()

            if not price:
                continue
            try:
                groups[key]["prices"].append(float(price))
            except ValueError:
                continue

            if area:
                try:
                    groups[key]["areas"].append(float(area))
                except ValueError:
                    pass

            if lease:
                try:
                    groups[key]["leases"].append(float(lease))
                except ValueError:
                    pass

            groups[key]["year"] = row.get("year", "")

    print(f"Aggregating {len(groups)} town-quarter-flat_type groups...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()

        for (town, quarter, flat_type), data in sorted(groups.items()):
            prices = data["prices"]
            areas = data["areas"]
            leases = data["leases"]

            writer.writerow({
                "town": town,
                "quarter": quarter,
                "year": data["year"],
                "flat_type": flat_type,
                "median_resale_price": round(median(prices), 2),
                "mean_resale_price": round(sum(prices) / len(prices), 2),
                "median_floor_area_sqm": round(median(areas), 2) if areas else "",
                "mean_floor_area_sqm": round(sum(areas) / len(areas), 2) if areas else "",
                "transaction_count": len(prices),
                "median_remaining_lease_years": round(median(leases), 2) if leases else "",
                "mean_remaining_lease_years": round(sum(leases) / len(leases), 2) if leases else "",
            })

    print(f"Aggregated -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()