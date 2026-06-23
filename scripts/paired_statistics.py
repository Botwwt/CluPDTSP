#!/usr/bin/env python
"""Compute paired CAADRL-versus-baseline statistics from aligned raw CSVs."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats


KEY_FIELDS = ("dataset", "size", "distribution", "decode", "seed")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def bootstrap_ci(values, seed, repetitions):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caadrl", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    caadrl_rows = {int(row["instance_id"]): row for row in read_rows(args.caadrl)}
    baseline_rows = {int(row["instance_id"]): row for row in read_rows(args.baseline)}
    if set(caadrl_rows) != set(baseline_rows):
        missing_left = sorted(set(baseline_rows).difference(caadrl_rows))
        missing_right = sorted(set(caadrl_rows).difference(baseline_rows))
        raise ValueError(f"Instance sets differ: missing_caadrl={missing_left[:5]} missing_baseline={missing_right[:5]}")

    first_id = min(caadrl_rows)
    first_caadrl = caadrl_rows[first_id]
    first_baseline = baseline_rows[first_id]
    for field in KEY_FIELDS:
        if first_caadrl[field] != first_baseline[field]:
            raise ValueError(f"Metadata mismatch for {field}: {first_caadrl[field]} != {first_baseline[field]}")

    gaps = []
    wins = ties = losses = 0
    for instance_id in sorted(caadrl_rows):
        left = caadrl_rows[instance_id]
        right = baseline_rows[instance_id]
        for field in KEY_FIELDS:
            if left[field] != right[field]:
                raise ValueError(f"Metadata mismatch for instance {instance_id}, field {field}")
        caadrl_objective = float(left["objective"])
        baseline_objective = float(right["objective"])
        if baseline_objective == 0:
            raise ZeroDivisionError(f"Baseline objective is zero for instance {instance_id}")
        gaps.append((caadrl_objective - baseline_objective) / baseline_objective * 100.0)
        delta = caadrl_objective - baseline_objective
        if delta < -args.tie_tolerance:
            wins += 1
        elif delta > args.tie_tolerance:
            losses += 1
        else:
            ties += 1

    gap_array = np.asarray(gaps, dtype=float)
    ci_low, ci_high = bootstrap_ci(gap_array, args.seed, args.bootstrap)
    ttest = stats.ttest_1samp(gap_array, popmean=0.0)
    std_gap = gap_array.std(ddof=1) if len(gap_array) > 1 else 0.0
    effect_size = float(gap_array.mean() / std_gap) if std_gap > 0 else 0.0
    row = {
        **{field: first_caadrl[field] for field in KEY_FIELDS},
        "comparator": f"CAADRL vs {args.baseline_label}",
        "n": len(gap_array),
        "mean_paired_gap_percent": float(gap_array.mean()),
        "ci95_low_percent": float(ci_low),
        "ci95_high_percent": float(ci_high),
        "p_value": float(ttest.pvalue),
        "effect_size_dz": effect_size,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "caadrl_raw": str(Path(args.caadrl).resolve()),
        "baseline_raw": str(Path(args.baseline).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
