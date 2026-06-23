#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-10 20 40 80}"
EPOCHS="${EPOCHS:-800}"
TEST_SIZE="${TEST_SIZE:-10000}"
TEST_SEED="${TEST_SEED:-10000}"
CLUSTER_STD="${CLUSTER_STD:-0.1}"
PROFILE_SIZE="${PROFILE_SIZE:-256}"
WARMUP="${WARMUP:-20}"
RAW_DIR="${RAW_DIR:-${ROOT_DIR}/results/profiling/raw}"

cd "${ROOT_DIR}"
mkdir -p "${RAW_DIR}"

for size in ${SIZES}; do
  model="$(find checkpoints/heter -path "*/pdp_${size}/*/epoch-$((EPOCHS - 1)).pt" -type f | sort | tail -n 1)"
  [[ -n "${model}" ]] || { echo "missing Heter PDP${size} checkpoint" >&2; exit 1; }
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
      output="${RAW_DIR}/heter_pdp${size}_${distribution}_${label}.csv"
      if [[ -f "${output}" ]]; then
        echo "reuse existing ${output}"
        continue
      fi
      DATASET="${dataset}" \
      MODEL="${model}" \
      SIZE="${size}" \
      DISTRIBUTION="${distribution}" \
      DECODE="${decode}" \
      WIDTH="${width}" \
      SEED="${TEST_SEED}" \
      PROFILE_SIZE="${PROFILE_SIZE}" \
      WARMUP="${WARMUP}" \
      OUTPUT="${output}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/profile_heter.sh
    done
  done
done

"${PYTHON_BIN}" scripts/summarize_profiling.py \
  --input-dir "${RAW_DIR}" \
  --csv-output results/profiling/heter_runtime_memory.csv \
  --tex-output results/profiling/heter_runtime_memory.tex
