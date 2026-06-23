#!/usr/bin/env python
"""Strictly validate an instance-level evaluation CSV before it is summarized."""

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_FIELDS = (
    "instance_id",
    "dataset",
    "size",
    "distribution",
    "decode",
    "seed",
    "objective",
    "route",
    "runtime_sec",
    "checkpoint",
    "command",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in REQUIRED_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"{raw_path}: missing columns {missing}")
        rows = list(reader)

    if len(rows) != args.expected_n:
        raise ValueError(f"{raw_path}: expected {args.expected_n} rows, found {len(rows)}")
    expected_ids = list(range(args.expected_n))
    actual_ids = [int(row["instance_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"{raw_path}: instance IDs are not exactly 0..{args.expected_n - 1}")

    expected_metadata = {
        "size": str(args.size),
        "distribution": args.distribution,
        "decode": args.decode,
        "seed": str(args.seed),
    }
    for index, row in enumerate(rows):
        for field, expected in expected_metadata.items():
            if row[field] != expected:
                raise ValueError(f"{raw_path}: row {index} has {field}={row[field]!r}, expected {expected!r}")
        if not row["dataset"] or not row["checkpoint"] or not row["command"]:
            raise ValueError(f"{raw_path}: row {index} has an empty provenance field")
        objective = float(row["objective"])
        runtime = float(row["runtime_sec"])
        if not math.isfinite(objective) or not math.isfinite(runtime) or runtime < 0:
            raise ValueError(f"{raw_path}: row {index} has invalid objective or runtime")
        route = json.loads(row["route"])
        if not isinstance(route, list) or not all(isinstance(node, int) for node in route):
            raise ValueError(f"{raw_path}: row {index} has an invalid route")

    print(f"VALID {raw_path}: {len(rows)} rows, PDP{args.size} {args.distribution} {args.decode}")


if __name__ == "__main__":
    main()
