#!/usr/bin/env python
"""Evaluate CAADRL/ablation checkpoints and write instance-level raw CSV."""

import argparse
import csv
import json
import os
import pickle
import shlex
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from CVRPEnv import CVRPEnv
from CVRPModel_training import CVRPModel
from experiment import ABLATION_CHOICES, apply_ablation


def load_dataset(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"], payload.get("scale_factor", 1.0)
    return payload, 1.0


def tensors_from_instance(instance, device):
    depot_coords, node_coords = instance[0], instance[1]
    depot_xy = torch.tensor([depot_coords], dtype=torch.float32, device=device).unsqueeze(0)
    node_xy = torch.tensor(node_coords, dtype=torch.float32, device=device).unsqueeze(0)
    problem_size = node_xy.size(1)
    if problem_size % 2 != 0:
        raise ValueError(f"Problem size must be even, got {problem_size}")
    num_pairs = problem_size // 2

    pickup_clusters = torch.full((1, num_pairs), 1, dtype=torch.long, device=device)
    delivery_clusters = torch.full((1, num_pairs), 2, dtype=torch.long, device=device)
    cluster_list = torch.cat([pickup_clusters, delivery_clusters], dim=1)

    pairing = torch.zeros(1, problem_size + 1, dtype=torch.long, device=device)
    pickup_indices = torch.arange(1, num_pairs + 1, device=device)
    delivery_indices = torch.arange(num_pairs + 1, problem_size + 1, device=device)
    pairing[:, pickup_indices] = delivery_indices
    pairing[:, delivery_indices] = pickup_indices

    node_demand = torch.zeros(size=(1, problem_size), device=device, dtype=torch.float32)
    return depot_xy, node_xy, node_demand, cluster_list, pairing


def tensors_from_instances(instances, device):
    depot_xy = torch.tensor([[instance[0]] for instance in instances], dtype=torch.float32, device=device)
    node_xy = torch.tensor([instance[1] for instance in instances], dtype=torch.float32, device=device)
    problem_size = node_xy.size(1)
    if problem_size % 2 != 0:
        raise ValueError(f"Problem size must be even, got {problem_size}")
    batch_size = node_xy.size(0)
    num_pairs = problem_size // 2

    pickup_clusters = torch.full((batch_size, num_pairs), 1, dtype=torch.long, device=device)
    delivery_clusters = torch.full((batch_size, num_pairs), 2, dtype=torch.long, device=device)
    cluster_list = torch.cat([pickup_clusters, delivery_clusters], dim=1)

    pairing = torch.zeros(batch_size, problem_size + 1, dtype=torch.long, device=device)
    pickup_indices = torch.arange(1, num_pairs + 1, device=device)
    delivery_indices = torch.arange(num_pairs + 1, problem_size + 1, device=device)
    pairing[:, pickup_indices] = delivery_indices
    pairing[:, delivery_indices] = pickup_indices

    node_demand = torch.zeros(size=(batch_size, problem_size), device=device, dtype=torch.float32)
    return depot_xy, node_xy, node_demand, cluster_list, pairing


def build_model_params(ablation, decode, width):
    eval_type = "greedy" if decode == "greedy" else "sampling"
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "use_learned_gate": True,
        "eval_type": eval_type,
    }
    return apply_ablation(model_params, ablation, disable_gate=False)


def load_model(checkpoint_path, model_params, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CVRPModel(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_route(route):
    route = [int(x) for x in route]
    if not route or route[0] != 0:
        route = [0] + route
    return route


def evaluate_instance(model, env, instance, device, scale):
    depot_xy, node_xy, node_demand, cluster_list, pairing = tensors_from_instance(instance, device)
    env.use_saved_problems(depot_xy, node_xy, node_demand, cluster_list, pairing)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        env.load_problems(1)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    best_reward, best_idx = reward.max(dim=1)
    route = env.selected_node_list[0, best_idx.item(), :].detach().cpu().tolist()
    objective = -best_reward.item() * scale
    return objective, normalize_route(route), duration


def evaluate_batch(model, env, instances, device, scale):
    depot_xy, node_xy, node_demand, cluster_list, pairing = tensors_from_instances(instances, device)
    batch_size = node_xy.size(0)
    env.use_saved_problems(depot_xy, node_xy, node_demand, cluster_list, pairing)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        env.load_problems(batch_size)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    best_reward, best_idx = reward.max(dim=1)
    selected_routes = env.selected_node_list.detach().cpu()
    results = []
    per_instance_runtime = duration / batch_size
    for row_idx in range(batch_size):
        route = selected_routes[row_idx, best_idx[row_idx].item(), :].tolist()
        objective = -best_reward[row_idx].item() * scale
        results.append((objective, normalize_route(route), per_instance_runtime))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--ablation", required=True, choices=ABLATION_CHOICES)
    parser.add_argument("--variant-label", default=None)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--distribution", required=True, choices=["clustered", "uniform", "li_lim_relaxed"])
    parser.add_argument("--decode", required=True, choices=["greedy", "sampling"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cuda-device-num", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device(f"cuda:{args.cuda_device_num}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)

    pomo_size = 1 if args.decode == "greedy" else args.width
    if args.decode == "sampling" and pomo_size <= 0:
        raise ValueError("--width must be positive for sampling")
    env = CVRPEnv(problem_size=0, pomo_size=pomo_size, device=device)
    model_params = build_model_params(args.ablation, args.decode, args.width)
    model = load_model(args.checkpoint, model_params, device)

    dataset, scale = load_dataset(args.dataset)
    selected = dataset[args.offset: args.offset + args.num_instances]
    dataset_name = args.dataset_name or Path(args.dataset).stem
    decode_label = f"Sampling-{args.width}" if args.decode == "sampling" else "Greedy"
    variant = args.variant_label or args.ablation
    command = " ".join(shlex.quote(part) for part in sys.argv)

    fields = [
        "method",
        "variant",
        "instance_id",
        "dataset",
        "size",
        "distribution",
        "decode",
        "seed",
        "objective",
        "route",
        "runtime_sec",
        "checkpoint",
        "command",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_suffix(output.suffix + ".partial")
    batch_size = max(1, args.batch_size)
    with partial_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for batch_start in range(0, len(selected), batch_size):
            batch = selected[batch_start: batch_start + batch_size]
            if batch_size == 1:
                batch_results = [evaluate_instance(model, env, batch[0], device, scale)]
            else:
                batch_results = evaluate_batch(model, env, batch, device, scale)
            for local_idx, (objective, route, runtime_sec) in enumerate(batch_results):
                idx = batch_start + local_idx
                writer.writerow(
                    {
                        "method": "CAADRL",
                        "variant": variant,
                        "instance_id": args.offset + idx,
                        "dataset": dataset_name,
                        "size": args.size,
                        "distribution": args.distribution,
                        "decode": decode_label,
                        "seed": args.seed,
                        "objective": objective,
                        "route": json.dumps(route, separators=(",", ":")),
                        "runtime_sec": runtime_sec,
                        "checkpoint": os.path.abspath(args.checkpoint),
                        "command": command,
                    }
                )
            if (batch_start + len(batch)) % 500 == 0 or batch_start + len(batch) == len(selected):
                print(f"evaluated {batch_start + len(batch)}/{len(selected)}")
    os.replace(partial_output, output)
    print(f"wrote {output} ({len(selected)} rows)")


if __name__ == "__main__":
    main()
