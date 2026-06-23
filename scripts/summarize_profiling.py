#!/usr/bin/env python
"""Combine per-configuration profiling rows into CSV and LaTeX artifacts."""

import argparse
import csv
from pathlib import Path


FIELDS = [
    "method", "size", "distribution", "decode", "seed", "n", "warmup",
    "checkpoint", "peak_gpu_allocated_gb", "peak_gpu_reserved_gb", "cpu_ram_gb",
    "latency_per_instance_sec", "throughput_inst_per_sec", "p95_latency_sec",
    "total_runtime_sec",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--tex-output", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda row: (int(row["size"]), row["distribution"], row["decode"]))

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    tex_output = Path(args.tex_output)
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    with tex_output.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllr rrrr}\\toprule\n")
        f.write("Method & Size & Distribution & Decode & GPU alloc. & GPU reserved & Latency (s) & Throughput \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['method']} & {row['size']} & {row['distribution']} & {row['decode']} & "
                f"{float(row['peak_gpu_allocated_gb']):.3f} & "
                f"{float(row['peak_gpu_reserved_gb']):.3f} & "
                f"{float(row['latency_per_instance_sec']):.6f} & "
                f"{float(row['throughput_inst_per_sec']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\\end{tabular}\n")
    print(f"wrote {csv_output} and {tex_output}")


if __name__ == "__main__":
    main()
