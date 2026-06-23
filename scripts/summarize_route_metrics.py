#!/usr/bin/env python
"""Summarize route-level metrics from one or more instance-level CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "inter_cluster_switch_count",
    "intra_cluster_edge_ratio",
    "cluster_separability_index",
)
BASE_GROUP_FIELDS = ("dataset", "size", "distribution", "decode", "seed")
OPTIONAL_GROUP_FIELDS = ("method", "variant")


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Route-metric CSV files")
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--tex-output", required=True)
    args = parser.parse_args()

    rows = []
    available_fields = set()
    for input_path in args.inputs:
        with open(input_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(METRICS).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing route metrics in {input_path}")
            for row in reader:
                rows.append(row)
                available_fields.update(row)
    if not rows:
        raise ValueError("No route-metric rows found")

    group_fields = [
        field for field in OPTIONAL_GROUP_FIELDS + BASE_GROUP_FIELDS if field in available_fields
    ]
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        for metric in METRICS:
            grouped[key][metric].append(float(row[metric]))

    summary_rows = []
    for key, values in sorted(grouped.items()):
        summary = dict(zip(group_fields, key))
        summary["n"] = len(values[METRICS[0]])
        for metric in METRICS:
            summary[f"mean_{metric}"] = mean(values[metric])
            summary[f"std_{metric}"] = sample_std(values[metric])
        summary_rows.append(summary)

    columns = group_fields + [
        "n",
        "mean_inter_cluster_switch_count",
        "std_inter_cluster_switch_count",
        "mean_intra_cluster_edge_ratio",
        "std_intra_cluster_edge_ratio",
        "mean_cluster_separability_index",
        "std_cluster_separability_index",
    ]
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    tex_output = Path(args.tex_output)
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    with tex_output.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(group_fields) + "rrrr}\\toprule\n")
        headers = [field.replace("_", " ").title() for field in group_fields]
        headers += ["N", "Switches", "Intra ratio", "Separability"]
        handle.write(" & ".join(headers) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in summary_rows:
            values = [str(row[field]) for field in group_fields]
            values += [
                str(row["n"]),
                f"{float(row['mean_inter_cluster_switch_count']):.3f}",
                f"{float(row['mean_intra_cluster_edge_ratio']):.4f}",
                f"{float(row['mean_cluster_separability_index']):.4f}",
            ]
            handle.write(" & ".join(values) + " \\\\\n")
        handle.write("\\bottomrule\\end{tabular}\n")
    print(f"wrote {csv_output} and {tex_output}")


if __name__ == "__main__":
    main()
