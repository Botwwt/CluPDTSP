#!/usr/bin/env python
"""Convert Li & Lim PDPTW-style files to relaxed single-vehicle PDP pickles.

The relaxed benchmark keeps depot and pickup-delivery coordinates, ignores time
windows, capacity, service times, and vehicle count, and reports only route
length and inference time downstream. It must not be compared directly with
PDPTW best-known solutions.
"""

import argparse
import csv
import pickle
import re
from pathlib import Path


def numeric_rows(path):
    rows = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue
        # Li & Lim instance headers may be numeric (for example, vehicle count,
        # capacity, and a type flag). Customer records have the full nine-column
        # PDPTW layout, so exclude numeric headers before inferring pairs.
        if len(values) >= 9:
            rows.append(values)
    return rows


def as_int(value):
    return int(round(value))


def parse_instance(path):
    rows = numeric_rows(path)
    if not rows:
        raise ValueError(f"No numeric customer rows found in {path}")

    coords = {as_int(row[0]): (float(row[1]), float(row[2])) for row in rows}
    depot_id = 0 if 0 in coords else as_int(rows[0][0])
    depot = coords[depot_id]

    pairs = []
    seen = set()
    for row in rows:
        node_id = as_int(row[0])
        if node_id == depot_id or len(row) < 9:
            continue
        pickup_ref = as_int(row[-2])
        delivery_ref = as_int(row[-1])
        if delivery_ref > 0 and delivery_ref in coords:
            pair = (node_id, delivery_ref)
        elif pickup_ref > 0 and pickup_ref in coords:
            pair = (pickup_ref, node_id)
        else:
            continue
        if pair[0] != depot_id and pair[1] != depot_id and pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if not pairs:
        raise ValueError(
            f"Could not recover pickup--delivery sibling pairs from {path}. "
            "Refusing to fabricate pairs from customer order."
        )

    pickups = [coords[pickup_id] for pickup_id, _ in pairs]
    deliveries = [coords[delivery_id] for _, delivery_id in pairs]
    return {
        "data": [(list(depot), [list(xy) for xy in pickups + deliveries])],
        "source": str(path),
        "num_pairs": len(pairs),
        "pairs": pairs,
        "relaxation": "single_vehicle_distance_only_ignore_time_windows_capacity_vehicle_count",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-dir", default="data/li_lim_relaxed")
    parser.add_argument("--summary", default="results/summary/li_lim_relaxed_manifest.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for input_path in args.inputs:
        source = Path(input_path)
        converted = parse_instance(source)
        out_path = out_dir / f"{source.stem}_relaxed_pdp.pkl"
        if out_path.exists() and not args.force:
            print(f"reuse existing {out_path}")
        else:
            with out_path.open("wb") as f:
                pickle.dump(converted["data"], f, pickle.HIGHEST_PROTOCOL)
            print(f"wrote {out_path}")
        summary_rows.append(
            {
                "source": str(source),
                "output": str(out_path),
                "num_pairs": converted["num_pairs"],
                "relaxation": converted["relaxation"],
            }
        )

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "output", "num_pairs", "relaxation"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
