#!/usr/bin/env python
"""Normalize legacy Heter raw CSVs without re-running model inference.

The initial Heter evaluator preserved the upstream batch wall-clock value on
every row of a greedy batch and omitted the leading depot from stored routes.
This utility archives the untouched source CSV, adds explicit provenance fields,
normalizes depot-delimited routes, and distributes a greedy batch time across
the exact number of rows in that batch. It never changes objectives or invokes
a checkpoint.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


def normalized_route(serialized):
    route = [int(node) for node in json.loads(serialized)]
    if not route or route[0] != 0:
        route.insert(0, 0)
    if route[-1] != 0:
        route.append(0)
    return json.dumps(route, separators=(",", ":"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-output", required=True)
    parser.add_argument("--greedy-batch-size", type=int, default=1024)
    parser.add_argument("--expected-n", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    legacy_output = Path(args.legacy_output)
    if source.resolve() != legacy_output.resolve():
        legacy_output.parent.mkdir(parents=True, exist_ok=True)
        if not legacy_output.exists():
            shutil.copy2(source, legacy_output)

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = reader.fieldnames or []
    if "runtime_protocol" in original_fields:
        if source.resolve() == output.resolve():
            print(f"reuse already-normalized {source}")
            return
        raise ValueError(f"{source} is already normalized; refusing to transform it again")
    if len(rows) != args.expected_n:
        raise ValueError(f"{source}: expected {args.expected_n} rows, found {len(rows)}")
    ids = [int(row["instance_id"]) for row in rows]
    if ids != list(range(args.expected_n)):
        raise ValueError(f"{source}: instance IDs must be exactly 0..{args.expected_n - 1}")
    decode_values = {row["decode"] for row in rows}
    if len(decode_values) != 1:
        raise ValueError(f"{source}: mixed decode values {decode_values}")
    decode = next(iter(decode_values))

    for row in rows:
        row["method"] = "Heter"
        row["variant"] = "baseline"
        row["route"] = normalized_route(row["route"])
        row["legacy_runtime_sec"] = row["runtime_sec"]
        row["legacy_source"] = str(legacy_output)
        row["postprocess_command"] = "normalize_heter_raw.py"

    if decode == "Greedy":
        for start in range(0, len(rows), args.greedy_batch_size):
            batch = rows[start:start + args.greedy_batch_size]
            reported = [float(row["legacy_runtime_sec"]) for row in batch]
            batch_duration = sum(reported) / len(reported)
            for row in batch:
                row["runtime_sec"] = repr(batch_duration / len(batch))
                row["runtime_protocol"] = "legacy_batch_duration_divided_by_exact_batch_size"
    else:
        for row in rows:
            row["runtime_protocol"] = "legacy_sampling_batch_size_1_unsynchronized"

    extra_fields = [
        "method", "variant", "legacy_runtime_sec", "legacy_source",
        "postprocess_command", "runtime_protocol",
    ]
    fieldnames = extra_fields + [field for field in original_fields if field not in extra_fields]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"normalized {source} -> {output}; archived original at {legacy_output}")


if __name__ == "__main__":
    main()
