#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-10 20 40 80}"
TEST_SIZE="${TEST_SIZE:-10000}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-1024}"

cd "${ROOT_DIR}"
for size in ${SIZES}; do
  for distribution in clustered uniform; do
    for label in greedy sampling_1280 sampling_12800; do
      raw="results/raw/heter/heter_pdp${size}_${distribution}_${label}.csv"
      legacy="results/raw/heter_legacy/heter_pdp${size}_${distribution}_${label}.csv"
      [[ -f "${raw}" ]] || { echo "missing ${raw}" >&2; exit 1; }
      "${PYTHON_BIN}" scripts/normalize_heter_raw.py \
        --input "${raw}" \
        --output "${raw}" \
        --legacy-output "${legacy}" \
        --greedy-batch-size "${GREEDY_BATCH_SIZE}" \
        --expected-n "${TEST_SIZE}"
    done
  done
done

PYTHON_BIN="${PYTHON_BIN}" bash scripts/validate_heter_main_grid.sh
"${PYTHON_BIN}" scripts/summarize_raw_results.py \
  "results/raw/heter/*.csv" \
  --csv-output results/summary/heter_main_results.csv \
  --tex-output results/summary/heter_main_results.tex
