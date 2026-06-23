#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-10 20 40 80}"
TEST_SIZE="${TEST_SIZE:-10000}"
TEST_SEED="${TEST_SEED:-10000}"

cd "${ROOT_DIR}"
for size in ${SIZES}; do
  for distribution in clustered uniform; do
    for label in greedy sampling_1280 sampling_12800; do
      case "${label}" in
        greedy) decode="Greedy" ;;
        sampling_1280) decode="Sampling-1280" ;;
        sampling_12800) decode="Sampling-12800" ;;
      esac
      "${PYTHON_BIN}" scripts/validate_raw_results.py \
        --raw "results/raw/heter/heter_pdp${size}_${distribution}_${label}.csv" \
        --expected-n "${TEST_SIZE}" \
        --size "${size}" \
        --distribution "${distribution}" \
        --decode "${decode}" \
        --seed "${TEST_SEED}"
    done
  done
done
