#!/usr/bin/env python
"""Compute route-level mechanism metrics and optional route visualization."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def route_with_start(route):
    route = [int(x) for x in route]
    if not route or route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route.append(0)
    return route


def cluster_id(node, size):
    if node == 0:
        return 0
    if 1 <= node <= size // 2:
        return 1
    return 2


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def separability(node_xy):
    pickups = np.array(node_xy[: len(node_xy) // 2], dtype=float)
    deliveries = np.array(node_xy[len(node_xy) // 2 :], dtype=float)
    pickup_center = pickups.mean(axis=0)
    delivery_center = deliveries.mean(axis=0)
    def diameter(points):
        if len(points) < 2:
            return 0.0
        deltas = points[:, None, :] - points[None, :, :]
        return float(np.sqrt(np.sum(deltas * deltas, axis=-1)).max())
    pickup_diam = diameter(pickups)
    delivery_diam = diameter(deliveries)
    return float(np.linalg.norm(pickup_center - delivery_center) / (pickup_diam + delivery_diam + 1e-12))


def compute_metrics(route, node_xy, size):
    route = route_with_start(route)
    transitions = list(zip(route[:-1], route[1:]))
    switch_count = 0
    same_cluster_edges = 0
    non_depot_edges = 0
    for left, right in transitions:
        left_cluster = cluster_id(left, size)
        right_cluster = cluster_id(right, size)
        if left_cluster != right_cluster:
            switch_count += 1
        if left_cluster != 0 and right_cluster != 0:
            non_depot_edges += 1
            if left_cluster == right_cluster:
                same_cluster_edges += 1
    intra_ratio = same_cluster_edges / non_depot_edges if non_depot_edges else 0.0
    return switch_count, intra_ratio, separability(node_xy)


def plot_route(instance, route, output_png, output_pdf, title):
    depot, node_xy = instance
    coords = np.array([depot] + node_xy, dtype=float)
    route = route_with_start(route)
    pickups = coords[1 : 1 + (len(coords) - 1) // 2]
    deliveries = coords[1 + (len(coords) - 1) // 2 :]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pickups[:, 0], pickups[:, 1], c="#377eb8", label="pickup", s=26)
    ax.scatter(deliveries[:, 0], deliveries[:, 1], c="#e41a1c", label="delivery", s=26)
    ax.scatter([depot[0]], [depot[1]], c="#111111", label="depot", marker="s", s=55)
    for left, right in zip(route[:-1], route[1:]):
        start = coords[left]
        end = coords[right]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#4d4d4d", "alpha": 0.75},
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--variant-label", default=None)
    parser.add_argument("--figure-instance", type=int, default=None)
    parser.add_argument("--figure-png", default="figures/route_visualization_caadr_vs_heter.png")
    parser.add_argument("--figure-pdf", default="figures/route_visualization_caadr_vs_heter.pdf")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_row = None

    partial_output = output.with_suffix(output.suffix + ".partial")
    with open(args.raw, newline="", encoding="utf-8") as f_in, partial_output.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames)
        for name, value in (("method", args.method_label), ("variant", args.variant_label)):
            if value is not None and name not in fieldnames:
                fieldnames.insert(0, name)
        fieldnames += [
            "inter_cluster_switch_count",
            "intra_cluster_edge_ratio",
            "cluster_separability_index",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if args.method_label is not None:
                row["method"] = args.method_label
            if args.variant_label is not None:
                row["variant"] = args.variant_label
            instance_id = int(row["instance_id"])
            size = int(row["size"])
            route = json.loads(row["route"])
            depot, node_xy = dataset[instance_id]
            switch_count, intra_ratio, sep = compute_metrics(route, node_xy, size)
            row.update(
                {
                    "inter_cluster_switch_count": switch_count,
                    "intra_cluster_edge_ratio": intra_ratio,
                    "cluster_separability_index": sep,
                }
            )
            writer.writerow(row)
            if args.figure_instance is not None and instance_id == args.figure_instance:
                figure_row = row

    partial_output.replace(output)

    if args.figure_instance is not None and figure_row is not None:
        route = json.loads(figure_row["route"])
        plot_route(
            dataset[int(figure_row["instance_id"])],
            route,
            args.figure_png,
            args.figure_pdf,
            f"Heter route, instance {figure_row['instance_id']}",
        )
        print(f"wrote {args.figure_png} and {args.figure_pdf}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
