#!/usr/bin/env python
"""Summarize raw instance-level CSV files into CSV and LaTeX tables."""

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path


BASE_GROUP_FIELDS = ["dataset", "size", "distribution", "decode", "seed"]
OPTIONAL_GROUP_FIELDS = ["method", "variant"]


def read_rows(patterns):
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["_source"] = path
                    yield row


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def write_tex(rows, output, group_fields):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        col_spec = "l" * len(group_fields) + "rrrr"
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\\toprule\n")
        headers = [field.replace("_", " ").title() for field in group_fields]
        f.write(" & ".join(headers + ["N", "Mean", "Std.", "Avg. time"]) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            group_values = [str(row[field]) for field in group_fields]
            f.write(
                " & ".join(group_values)
                + f" & {row['n']} & {float(row['mean_objective']):.6f} & "
                f"{float(row['std_objective']):.6f} & "
                f"{float(row['avg_runtime_sec']):.6f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--tex-output", required=True)
    args = parser.parse_args()

    rows = list(read_rows(args.inputs))
    available_fields = set()
    for row in rows:
        available_fields.update(row.keys())
    group_fields = [
        field for field in OPTIONAL_GROUP_FIELDS + BASE_GROUP_FIELDS
        if field in available_fields
    ]
    groups = defaultdict(lambda: {"objective": [], "runtime": [], "sources": set()})
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups[key]["objective"].append(float(row["objective"]))
        groups[key]["runtime"].append(float(row["runtime_sec"]))
        groups[key]["sources"].add(row["_source"])

    out_rows = []
    for key, values in sorted(groups.items()):
        objectives = values["objective"]
        runtimes = values["runtime"]
        out_rows.append(
            {
                **dict(zip(group_fields, key)),
                "n": len(objectives),
                "mean_objective": mean(objectives),
                "std_objective": std(objectives),
                "avg_runtime_sec": mean(runtimes),
                "total_runtime_sec": sum(runtimes),
                "sources": ";".join(sorted(values["sources"])),
            }
        )

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = group_fields + [
        "n",
        "mean_objective",
        "std_objective",
        "avg_runtime_sec",
        "total_runtime_sec",
        "sources",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    write_tex(out_rows, args.tex_output, group_fields)
    print(f"wrote {csv_output} and {args.tex_output}")


if __name__ == "__main__":
    main()
