#!/usr/bin/env python
"""Render CAADRL and Heter routes for the same unified PDP instance."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_instance(path, instance_id):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    return data[instance_id]


def row_for_instance(path, instance_id):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["instance_id"]) == instance_id:
                return row
    raise ValueError(f"Instance {instance_id} not found in {path}")


def normalize_route(route):
    route = [int(node) for node in route]
    if not route or route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route.append(0)
    return route


def cluster_id(node, pair_count):
    if node == 0:
        return 0
    return 1 if node <= pair_count else 2


def draw(ax, depot, node_xy, route, title):
    coords = np.asarray([depot] + node_xy, dtype=float)
    pair_count = len(node_xy) // 2
    ax.scatter(coords[1:1 + pair_count, 0], coords[1:1 + pair_count, 1], c="#2878b5", s=24, label="pickup")
    ax.scatter(coords[1 + pair_count:, 0], coords[1 + pair_count:, 1], c="#d84a4a", s=24, label="delivery")
    ax.scatter(coords[0, 0], coords[0, 1], c="#222222", marker="s", s=54, label="depot")
    route = normalize_route(route)
    switches = 0
    for step, (left, right) in enumerate(zip(route[:-1], route[1:]), start=1):
        cross_cluster = cluster_id(left, pair_count) != cluster_id(right, pair_count)
        color = "#d95f02" if cross_cluster else "#555555"
        if cross_cluster:
            switches += 1
        ax.annotate("", xy=coords[right], xytext=coords[left], arrowprops={"arrowstyle": "->", "lw": 1.15 if cross_cluster else 0.7, "color": color, "alpha": 0.78})
        if cross_cluster:
            midpoint = (coords[left] + coords[right]) / 2
            ax.text(midpoint[0], midpoint[1], str(step), color=color, fontsize=7, ha="center", va="center")
    ax.set_title(f"{title}\ninter-cluster switches: {switches}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--caadrl-raw", required=True)
    parser.add_argument("--heter-raw", required=True)
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--png", required=True)
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()

    depot, node_xy = load_instance(args.dataset, args.instance_id)
    caadrl = row_for_instance(args.caadrl_raw, args.instance_id)
    heter = row_for_instance(args.heter_raw, args.instance_id)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    draw(axes[0], depot, node_xy, json.loads(caadrl["route"]), f"CAADRL, objective {float(caadrl['objective']):.4f}")
    draw(axes[1], depot, node_xy, json.loads(heter["route"]), f"Heter, objective {float(heter['objective']):.4f}")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#d95f02", lw=1.15, label="inter-cluster transition"))
    labels.append("inter-cluster transition")
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(f"PDP route comparison, instance {args.instance_id}")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    for output in (Path(args.png), Path(args.pdf)):
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=220 if output.suffix == ".png" else None)
    plt.close(figure)
    print(f"wrote {args.png} and {args.pdf}")


if __name__ == "__main__":
    main()
