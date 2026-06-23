#!/usr/bin/env python
"""Format paired-comparison CSV summaries as compact LaTeX tables."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llllrrrrr}\\toprule\n")
        f.write(
            "Dataset & Decode & Comparator & N & Gap (\\%) & 95\\% CI & $p$ & $d_z$ & W/T/L \\\\\n"
        )
        f.write("\\midrule\n")
        for row in rows:
            dataset = f"PDP{row['size']}-{row['distribution']}"
            gap = float(row["mean_paired_gap_percent"])
            low = float(row["ci95_low_percent"])
            high = float(row["ci95_high_percent"])
            p_value = float(row["p_value"])
            effect = float(row["effect_size_dz"])
            if p_value < 1e-4:
                p_text = "$<10^{-4}$"
            else:
                p_text = f"{p_value:.4f}"
            wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
            f.write(
                f"{dataset} & {row['decode']} & {row['comparator']} & {row['n']} & "
                f"{gap:.3f} & [{low:.3f}, {high:.3f}] & {p_text} & {effect:.3f} & {wtl} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
