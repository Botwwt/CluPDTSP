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
PROFILE_SIZE="${PROFILE_SIZE:-256}"
WARMUP="${WARMUP:-20}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/results/profiling/heter_runtime_memory.csv}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/profile_heter.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --size "${SIZE}" \
  --distribution "${DISTRIBUTION}" \
  --decode "${DECODE}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --profile-size "${PROFILE_SIZE}" \
  --warmup "${WARMUP}" \
  --force
