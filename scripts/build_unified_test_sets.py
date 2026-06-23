#!/usr/bin/env python
"""Build shared PDP test sets for CAADRL, Heter, and ablations.

The repository's original generator writes one single-instance pickle per seed.
This script preserves that per-instance seed order in one aggregate pickle so
baselines that expect a 10,000-instance file can consume the exact same cases.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def generate_clustered_instance(pdp_size, cluster_std):
    if pdp_size % 2 != 0:
        raise ValueError(f"pdp_size must be even, got {pdp_size}")
    num_pairs = pdp_size // 2
    depot_xy = [0.5, 0.5]
    pickup_center = np.array([0.25, 0.25])
    delivery_center = np.array([0.75, 0.75])
    pickups = pickup_center + np.random.randn(num_pairs, 2) * cluster_std
    deliveries = delivery_center + np.random.randn(num_pairs, 2) * cluster_std
    node_coords = np.concatenate((pickups, deliveries), axis=0)
    node_coords = np.clip(node_coords, 0, 1)
    return depot_xy, node_coords.tolist()


def generate_uniform_instance(pdp_size):
    depot_xy = np.random.uniform(size=(2,)).tolist()
    node_coords = np.random.uniform(size=(pdp_size, 2)).tolist()
    return depot_xy, node_coords


def generate_dataset(size, distribution, num_instances, seed, cluster_std):
    dataset = []
    for offset in range(num_instances):
        np.random.seed(seed + offset)
        if distribution == "clustered":
            dataset.append(generate_clustered_instance(size, cluster_std))
        elif distribution == "uniform":
            dataset.append(generate_uniform_instance(size))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    return dataset


def save_pickle(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--distributions", nargs="+", default=["clustered", "uniform"])
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--name", default="test")
    parser.add_argument("--cluster-std", type=float, default=0.1)
    parser.add_argument("--out-dir", default="data/pdp/unified")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for size in args.sizes:
        for distribution in args.distributions:
            suffix = (
                f"clustered_std{args.cluster_std}"
                if distribution == "clustered"
                else "uniform"
            )
            filename = (
                f"pdp{size}_{args.name}_{suffix}_seed{args.seed}_n{args.num_instances}.pkl"
            )
            out_path = out_dir / filename
            if out_path.exists() and not args.force:
                print(f"reuse existing {out_path}")
                continue
            dataset = generate_dataset(
                size=size,
                distribution=distribution,
                num_instances=args.num_instances,
                seed=args.seed,
                cluster_std=args.cluster_std,
            )
            save_pickle(dataset, out_path)
            print(f"wrote {out_path} ({len(dataset)} instances)")


if __name__ == "__main__":
    main()
