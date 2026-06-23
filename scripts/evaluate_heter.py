#!/usr/bin/env python
"""Evaluate the Heter baseline and write instance-level raw CSV results."""

import argparse
import csv
import importlib.util
import json
import os
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


def load_heter_eval(repo_root):
    heter_root = repo_root / "baselines" / "heter"
    sys.path.insert(0, str(heter_root))
    spec = importlib.util.spec_from_file_location("heter_eval", heter_root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_route(route):
    if route is None:
        return []
    if hasattr(route, "tolist"):
        route = route.tolist()
    route = [int(x) for x in route]
    if not route or route[0] != 0:
        route.insert(0, 0)
    if route[-1] != 0:
        route.append(0)
    return route


def adjust_sampling_args(args):
    if args.decode != "sampling":
        return
    if args.width <= 0:
        raise ValueError("--width must be positive for sampling")
    if args.width % args.max_calc_batch_size != 0:
        for candidate in [1280, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
            if args.width % candidate == 0:
                args.max_calc_batch_size = candidate
                break
    if args.width * args.eval_batch_size > args.max_calc_batch_size:
        args.eval_batch_size = 1


def evaluate_dataset(heter_eval, model, dataset, width, opts, device):
    """Evaluate greedy or sampled Heter routes with synchronized batch timing.

    The upstream ``_eval_dataset`` reports one batch duration beside every row in
    that batch.  That is useful for its original aggregate report, but it would
    over-count the total time in an instance-level CSV.  This implementation
    preserves the upstream decoding path and divides the synchronized batch
    duration across exactly the instances in that batch.
    """
    model.to(device)
    model.eval()
    model.set_decode_type(
        "greedy" if opts.decode_strategy == "greedy" else "sampling",
        temp=1.0,
    )

    results = []
    for batch in DataLoader(dataset, batch_size=opts.eval_batch_size):
        batch = heter_eval.move_to(batch, device)
        batch_size = len(batch["depot"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            if opts.decode_strategy == "greedy":
                if width != 0:
                    raise ValueError("Greedy Heter evaluation requires width=0")
                if batch_size > opts.max_calc_batch_size:
                    raise ValueError("eval batch is larger than max calculation batch size")
                batch_rep, iter_rep = 1, 1
            else:
                if width <= 0:
                    raise ValueError("Sampling Heter evaluation requires width > 0")
                if width * batch_size > opts.max_calc_batch_size:
                    if batch_size != 1 or width % opts.max_calc_batch_size != 0:
                        raise ValueError("invalid sampling batch/max-calculation configuration")
                    batch_rep, iter_rep = opts.max_calc_batch_size, width // opts.max_calc_batch_size
                else:
                    batch_rep, iter_rep = width, 1
            sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            ids = torch.arange(len(costs), dtype=torch.int64, device=costs.device)
            sequences, costs = heter_eval.get_best(
                sequences.detach().cpu().numpy(),
                costs.detach().cpu().numpy(),
                ids.detach().cpu().numpy(),
                batch_size,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_instance_duration = (time.perf_counter() - start) / batch_size
        for sequence, cost in zip(sequences, costs):
            route = heter_eval.np.trim_zeros(sequence).tolist() + [0]
            results.append((float(cost), normalize_route(route), per_instance_duration))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--distribution", required=True, choices=["clustered", "uniform", "li_lim_relaxed"])
    parser.add_argument("--decode", required=True, choices=["greedy", "sampling"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
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
    dataset = model.problem.make_dataset(
        filename=args.dataset,
        num_samples=args.val_size,
        offset=args.offset,
    )

    decode_strategy = "greedy" if args.decode == "greedy" else "sample"
    width = 0 if args.decode == "greedy" else args.width
    opts = SimpleNamespace(
        decode_strategy=decode_strategy,
        eval_batch_size=args.eval_batch_size,
        max_calc_batch_size=args.max_calc_batch_size,
        compress_mask=False,
        no_progress_bar=False,
    )

    results = evaluate_dataset(heter_eval, model, dataset, width, opts, device)

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or Path(args.dataset).stem
    command = " ".join(shlex.quote(part) for part in sys.argv)
    checkpoint = os.path.abspath(args.model)
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
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for local_idx, (cost, route, duration) in enumerate(results):
            writer.writerow(
                {
                    "method": "Heter",
                    "variant": "baseline",
                    "instance_id": args.offset + local_idx,
                    "dataset": dataset_name,
                    "size": args.size,
                    "distribution": args.distribution,
                    "decode": f"Sampling-{width}" if args.decode == "sampling" else "Greedy",
                    "seed": args.seed,
                    "objective": float(cost),
                    "route": json.dumps(normalize_route(route), separators=(",", ":")),
                    "runtime_sec": float(duration),
                    "checkpoint": checkpoint,
                    "command": command,
                }
            )
    print(f"wrote {output} ({len(results)} rows)")


if __name__ == "__main__":
    main()
