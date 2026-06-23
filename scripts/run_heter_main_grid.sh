#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-10 20 40 80}"
EPOCHS="${EPOCHS:-800}"
EPOCH_SIZE="${EPOCH_SIZE:-1280000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VAL_SIZE="${VAL_SIZE:-10000}"
TEST_SIZE="${TEST_SIZE:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-100}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
TEST_SEED="${TEST_SEED:-10000}"
CLUSTER_STD="${CLUSTER_STD:-0.1}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/build_unified_test_sets.py \
  --sizes ${SIZES} \
  --distributions uniform \
  --num-instances "${VAL_SIZE}" \
  --seed "${TRAIN_SEED}" \
  --name validation \
  --out-dir data/pdp/validation

"${PYTHON_BIN}" scripts/build_unified_test_sets.py \
  --sizes ${SIZES} \
  --distributions clustered uniform \
  --num-instances "${TEST_SIZE}" \
  --seed "${TEST_SEED}" \
  --cluster-std "${CLUSTER_STD}" \
  --name test \
  --out-dir data/pdp/unified

for size in ${SIZES}; do
  checkpoint="$(find checkpoints/heter -path "*/pdp_${size}/*/epoch-$((EPOCHS - 1)).pt" -type f 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${checkpoint}" ]]; then
    VAL_DATASET="${ROOT_DIR}/data/pdp/validation/pdp${size}_validation_uniform_seed${TRAIN_SEED}_n${VAL_SIZE}.pkl" \
    SIZE="${size}" \
    SEED="${TRAIN_SEED}" \
    EPOCHS="${EPOCHS}" \
    EPOCH_SIZE="${EPOCH_SIZE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    VAL_SIZE="${VAL_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS}" \
    RUN_NAME="heter_pdp${size}_rollout" \
    OUTPUT_DIR="${ROOT_DIR}/checkpoints/heter" \
    NO_TENSORBOARD=1 \
    bash scripts/run_heter_train.sh --no_progress_bar
    checkpoint="$(find checkpoints/heter -path "*/pdp_${size}/*/epoch-$((EPOCHS - 1)).pt" -type f | sort | tail -n 1)"
  fi
  for distribution in clustered uniform; do
    if [[ "${distribution}" == "clustered" ]]; then
      dataset="${ROOT_DIR}/data/pdp/unified/pdp${size}_test_clustered_std${CLUSTER_STD}_seed${TEST_SEED}_n${TEST_SIZE}.pkl"
    else
      dataset="${ROOT_DIR}/data/pdp/unified/pdp${size}_test_uniform_seed${TEST_SEED}_n${TEST_SIZE}.pkl"
    fi
    for label in greedy sampling_1280 sampling_12800; do
      case "${label}" in
        greedy) decode=greedy; width=0 ;;
        sampling_1280) decode=sampling; width=1280 ;;
        sampling_12800) decode=sampling; width=12800 ;;
      esac
      output="${ROOT_DIR}/results/raw/heter/heter_pdp${size}_${distribution}_${label}.csv"
      if [[ -f "${output}" ]]; then
        echo "reuse existing ${output}"
        continue
      fi
      DATASET="${dataset}" \
      MODEL="${checkpoint}" \
      SIZE="${size}" \
      DISTRIBUTION="${distribution}" \
      DECODE="${decode}" \
      WIDTH="${width}" \
      SEED="${TEST_SEED}" \
      VAL_SIZE="${TEST_SIZE}" \
      OUTPUT="${output}" \
      bash scripts/run_heter_eval.sh
    done
  done
done

"${PYTHON_BIN}" scripts/summarize_raw_results.py \
  "results/raw/heter/*.csv" \
  --csv-output results/summary/heter_main_results.csv \
  --tex-output results/summary/heter_main_results.tex
