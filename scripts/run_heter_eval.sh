#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?DATASET is required}"
MODEL="${MODEL:?MODEL is required}"
SIZE="${SIZE:?SIZE is required}"
DISTRIBUTION="${DISTRIBUTION:?DISTRIBUTION is required}"
DECODE="${DECODE:-greedy}"
WIDTH="${WIDTH:-0}"
SEED="${SEED:-10000}"
VAL_SIZE="${VAL_SIZE:-10000}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/results/raw/heter/heter_pdp${SIZE}_${DISTRIBUTION}_${DECODE}_${WIDTH}.csv}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/evaluate_heter.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --size "${SIZE}" \
  --distribution "${DISTRIBUTION}" \
  --decode "${DECODE}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --val-size "${VAL_SIZE}"
