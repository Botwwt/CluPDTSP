#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:?MODEL is required}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/data/li_lim_relaxed}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/raw/heter_li_lim_relaxed}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/results/summary/li_lim_relaxed_manifest.csv}"
DECODE="${DECODE:-greedy}"
WIDTH="${WIDTH:-0}"
SEED="${SEED:-10000}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

for dataset in "${INPUT_DIR}"/*_relaxed_pdp.pkl; do
  [[ -f "${dataset}" ]] || continue
  dataset_name="$(basename "${dataset}" .pkl)"
  output="${OUTPUT_DIR}/${dataset_name}_${DECODE}_${WIDTH}.csv"
  if [[ -f "${output}" ]]; then
    echo "reuse existing ${output}"
    continue
  fi
  size="$(${PYTHON_BIN} -c 'import pickle,sys; print(len(pickle.load(open(sys.argv[1], "rb"))[0][1]))' "${dataset}")"
  "${PYTHON_BIN}" scripts/evaluate_heter.py \
    --dataset "${dataset}" \
    --model "${MODEL}" \
    --output "${output}" \
    --dataset-name "${dataset_name}" \
    --size "${size}" \
    --distribution li_lim_relaxed \
    --decode "${DECODE}" \
    --width "${WIDTH}" \
    --seed "${SEED}" \
    --val-size 1 \
    --force
done

"${PYTHON_BIN}" scripts/summarize_li_lim_relaxed.py \
  --raw-dir "${OUTPUT_DIR}" \
  --manifest "${MANIFEST}" \
  --csv-output results/summary/heter_li_lim_relaxed.csv \
  --tex-output results/summary/heter_li_lim_relaxed.tex
