#!/usr/bin/env python
"""Summarize relaxed Li & Lim Heter evaluations without PDPTW comparisons."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def source_groups(manifest_path):
    groups = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            output = Path(row["output"]).stem
            groups[output] = Path(row["source"]).parent.name
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--tex-output", required=True)
    args = parser.parse_args()

    groups = source_groups(args.manifest)
    aggregate = defaultdict(lambda: {"objective": [], "runtime": [], "sources": []})
    for raw_path in sorted(Path(args.raw_dir).glob("*.csv")):
        with raw_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_group = groups.get(row["dataset"], "unknown")
                key = (source_group, row["decode"])
                aggregate[key]["objective"].append(float(row["objective"]))
                aggregate[key]["runtime"].append(float(row["runtime_sec"]))
                aggregate[key]["sources"].append(row["dataset"])

    rows = []
    for (source_group, decode), values in sorted(aggregate.items()):
        rows.append(
            {
                "source_group": source_group,
                "decode": decode,
                "n": len(values["objective"]),
                "mean_route_length": mean(values["objective"]),
                "std_route_length": sample_std(values["objective"]),
                "avg_inference_time_sec": mean(values["runtime"]),
                "total_inference_time_sec": sum(values["runtime"]),
                "sources": ";".join(sorted(values["sources"])),
                "scope": "single_vehicle_distance_only_relaxed_pdptw",
            }
        )

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "source_group", "decode", "n", "mean_route_length", "std_route_length",
        "avg_inference_time_sec", "total_inference_time_sec", "sources", "scope",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_output = Path(args.tex_output)
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    with tex_output.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrr}\\toprule\n")
        f.write("Source set & Decode & N & Mean route length & Std. & Avg. time (s) \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['source_group']} & {row['decode']} & {row['n']} & "
                f"{row['mean_route_length']:.6f} & {row['std_route_length']:.6f} & "
                f"{row['avg_inference_time_sec']:.6f} \\\\\n"
            )
        f.write("\\bottomrule\\end{tabular}\n")
    print(f"wrote {csv_output} and {tex_output}")


if __name__ == "__main__":
    main()
