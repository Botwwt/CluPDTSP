#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
HETER_GRID_PID="${HETER_GRID_PID:-}"
EPOCHS="${EPOCHS:-800}"

cd "${ROOT_DIR}"

if [[ -n "${HETER_GRID_PID}" ]] && kill -0 "${HETER_GRID_PID}" 2>/dev/null; then
  echo "waiting for Heter main grid PID ${HETER_GRID_PID}"
  while kill -0 "${HETER_GRID_PID}" 2>/dev/null; do
    sleep 60
  done
fi

echo "validating complete Heter main grid"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/validate_heter_main_grid.sh
"${PYTHON_BIN}" scripts/summarize_raw_results.py \
  "results/raw/heter/*.csv" \
  --csv-output results/summary/heter_main_results.csv \
  --tex-output results/summary/heter_main_results.tex

heter_model="$(find checkpoints/heter -path "*/pdp_80/*/epoch-$((EPOCHS - 1)).pt" -type f | sort | tail -n 1)"
[[ -n "${heter_model}" ]] || { echo "missing final Heter PDP80 checkpoint" >&2; exit 1; }

echo "running Li&Lim PDP-relaxed benchmark"
MODEL="${heter_model}" PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_heter_li_lim_relaxed.sh

echo "running Heter runtime and memory profiling"
EPOCHS="${EPOCHS}" PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_heter_profile_grid.sh

echo "running Wang-task CAADRL ablation grid"
PYTHON_BIN="${PYTHON_BIN}" CHECKPOINT_INTERVAL=100 bash scripts/run_ablation_grid.sh

echo "computing Heter mechanism metrics and available paired artifacts"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_heter_mechanism_analysis.sh

echo "WANG_POST_HETER_PIPELINE_COMPLETE"
