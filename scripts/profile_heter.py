#!/usr/bin/env python
"""Runtime and memory profiling for the Heter baseline."""

import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_heter import adjust_sampling_args, evaluate_dataset, load_heter_eval


def cpu_ram_gb():
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024 * 1024)
    except Exception:
        return 0.0


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--decode", required=True, choices=["greedy", "sampling"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--profile-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-calc-batch-size", type=int, default=10000)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    adjust_sampling_args(args)

    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")

    repo_root = Path(__file__).resolve().parents[1]
    heter_eval = load_heter_eval(repo_root)
    model, _ = heter_eval.load_model(args.model)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    decode_strategy = "greedy" if args.decode == "greedy" else "sample"
    width = 0 if args.decode == "greedy" else args.width
    opts = SimpleNamespace(
        decode_strategy=decode_strategy,
        eval_batch_size=args.eval_batch_size,
        max_calc_batch_size=args.max_calc_batch_size,
        compress_mask=False,
        no_progress_bar=True,
    )

    warmup_dataset = model.problem.make_dataset(
        filename=args.dataset,
        num_samples=max(1, args.warmup),
        offset=0,
    )
    evaluate_dataset(heter_eval, model, warmup_dataset, width, opts, device)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    profile_dataset = model.problem.make_dataset(
        filename=args.dataset,
        num_samples=args.profile_size,
        offset=0,
    )
    results = evaluate_dataset(heter_eval, model, profile_dataset, width, opts, device)
    total = sum(duration for _, _, duration in results)

    latencies = [float(duration) for _, _, duration in results]
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else 0.0
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if use_cuda else 0.0
    row = {
        "method": "Heter",
        "size": args.size,
        "distribution": args.distribution,
        "decode": f"Sampling-{width}" if args.decode == "sampling" else "Greedy",
        "seed": args.seed,
        "n": len(results),
        "warmup": args.warmup,
        "checkpoint": os.path.abspath(args.model),
        "peak_gpu_allocated_gb": allocated,
        "peak_gpu_reserved_gb": reserved,
        "cpu_ram_gb": cpu_ram_gb(),
        "latency_per_instance_sec": statistics.mean(latencies) if latencies else 0.0,
        "throughput_inst_per_sec": len(results) / total if total > 0 else 0.0,
        "p95_latency_sec": percentile(latencies, 0.95),
        "total_runtime_sec": total,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote profiling row to {output}")


if __name__ == "__main__":
    main()
