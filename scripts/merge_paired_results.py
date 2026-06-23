#!/usr/bin/env python
"""Join aligned CAADRL and Heter raw results without inventing missing pairs."""

import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("dataset", "size", "distribution", "decode", "seed")


def read_unique_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    mapped = {int(row["instance_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Duplicate instance IDs in {path}")
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caadrl", required=True)
    parser.add_argument("--heter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    caadrl = read_unique_rows(args.caadrl)
    heter = read_unique_rows(args.heter)
    if set(caadrl) != set(heter):
        raise ValueError("CAADRL and Heter raw files do not contain the same instance IDs")

    out_rows = []
    for instance_id in sorted(caadrl):
        caadrl_row, heter_row = caadrl[instance_id], heter[instance_id]
        for field in KEY_FIELDS:
            if caadrl_row.get(field) != heter_row.get(field):
                raise ValueError(
                    f"Metadata mismatch for instance {instance_id}: {field}="
                    f"{caadrl_row.get(field)!r} vs {heter_row.get(field)!r}"
                )
        caadrl_objective = float(caadrl_row["objective"])
        heter_objective = float(heter_row["objective"])
        if heter_objective == 0:
            raise ZeroDivisionError(f"Heter objective is zero for instance {instance_id}")
        out_rows.append(
            {
                "instance_id": instance_id,
                **{field: caadrl_row[field] for field in KEY_FIELDS},
                "caadrl_objective": caadrl_objective,
                "heter_objective": heter_objective,
                "caadrl_minus_heter": caadrl_objective - heter_objective,
                "caadrl_relative_improvement_percent":
                    (heter_objective - caadrl_objective) / heter_objective * 100.0,
                "caadrl_raw": str(Path(args.caadrl).resolve()),
                "heter_raw": str(Path(args.heter).resolve()),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {output} ({len(out_rows)} aligned instances)")


if __name__ == "__main__":
    main()
