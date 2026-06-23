#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZE="${SIZE:-20}"
SEED="${SEED:-1234}"
EPOCHS="${EPOCHS:-800}"
EPOCH_SIZE="${EPOCH_SIZE:-1280000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VAL_SIZE="${VAL_SIZE:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-heter_pdp${SIZE}_rollout}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/heter}"
VAL_DATASET="${VAL_DATASET:-}"
NO_TENSORBOARD="${NO_TENSORBOARD:-1}"

cd "${ROOT_DIR}/baselines/heter"

cmd=(
  "${PYTHON_BIN}" run.py
  --graph_size "${SIZE}"
  --baseline rollout
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --n_epochs "${EPOCHS}"
  --epoch_size "${EPOCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --val_size "${VAL_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --checkpoint_epochs "${CHECKPOINT_EPOCHS}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${VAL_DATASET}" ]]; then
  cmd+=(--val_dataset "${VAL_DATASET}")
fi

if [[ "${NO_TENSORBOARD}" == "1" ]]; then
  cmd+=(--no_tensorboard)
fi

cmd+=("$@")

printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
