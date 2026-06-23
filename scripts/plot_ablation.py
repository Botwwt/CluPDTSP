#!/usr/bin/env python
"""Create an ablation bar plot from summary CSV output."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary/ablation_results.csv")
    parser.add_argument("--decode", default="Greedy")
    parser.add_argument("--png", default="figures/ablation_barplot.png")
    parser.add_argument("--pdf", default="figures/ablation_barplot.pdf")
    args = parser.parse_args()

    rows = [
        row for row in read_rows(args.summary)
        if row.get("decode") == args.decode and row.get("distribution") == "clustered"
    ]
    if not rows:
        raise ValueError(f"No rows for decode={args.decode}")

    variants = sorted({row["variant"] for row in rows})
    sizes = sorted({int(row["size"]) for row in rows})
    width = 0.8 / max(1, len(variants))
    x_positions = list(range(len(sizes)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, variant in enumerate(variants):
        values = []
        for size in sizes:
            match = next(
                row for row in rows
                if row["variant"] == variant and int(row["size"]) == size
            )
            values.append(float(match["mean_objective"]))
        xs = [x + (offset - (len(variants) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=variant)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"PDP{size}" for size in sizes])
    ax.set_ylabel("Mean objective")
    ax.set_xlabel("Clustered test set")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    Path(args.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=220)
    fig.savefig(args.pdf)
    print(f"wrote {args.png} and {args.pdf}")


if __name__ == "__main__":
    main()
