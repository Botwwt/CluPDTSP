#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-10 20 40 80}"
TEST_SEED="${TEST_SEED:-10000}"
TEST_SIZE="${TEST_SIZE:-10000}"
CLUSTER_STD="${CLUSTER_STD:-0.1}"

cd "${ROOT_DIR}"
METRICS_DIR="results/stats/heter_route_metrics"
mkdir -p "${METRICS_DIR}"

for size in ${SIZES}; do
  for distribution in clustered uniform; do
    if [[ "${distribution}" == "clustered" ]]; then
      dataset="data/pdp/unified/pdp${size}_test_clustered_std${CLUSTER_STD}_seed${TEST_SEED}_n${TEST_SIZE}.pkl"
    else
      dataset="data/pdp/unified/pdp${size}_test_uniform_seed${TEST_SEED}_n${TEST_SIZE}.pkl"
    fi
    for label in greedy sampling_1280 sampling_12800; do
      raw="results/raw/heter/heter_pdp${size}_${distribution}_${label}.csv"
      output="${METRICS_DIR}/heter_pdp${size}_${distribution}_${label}.csv"
      [[ -f "${raw}" ]] || { echo "missing ${raw}" >&2; exit 1; }
      if [[ -f "${output}" ]]; then
        echo "reuse existing ${output}"
        continue
      fi
      "${PYTHON_BIN}" scripts/compute_route_metrics.py \
        --dataset "${dataset}" \
        --raw "${raw}" \
        --output "${output}" \
        --method-label Heter \
        --variant-label baseline
    done
  done
done

"${PYTHON_BIN}" scripts/summarize_route_metrics.py \
  ${METRICS_DIR}/*.csv \
  --csv-output results/stats/heter_route_metrics_summary.csv \
  --tex-output results/stats/heter_route_metrics_summary.tex

# Do not manufacture a CAADRL comparison: create paired improvements and a
# shared-instance route figure only when the matching CAADRL raw output exists.
CAADRL_RAW="results/raw/ablation/ablation_full_pdp40_clustered_greedy.csv"
HETER_RAW="results/raw/heter/heter_pdp40_clustered_greedy.csv"
if [[ -f "${CAADRL_RAW}" && -f "${HETER_RAW}" ]]; then
  "${PYTHON_BIN}" scripts/merge_paired_results.py \
    --caadrl "${CAADRL_RAW}" \
    --heter "${HETER_RAW}" \
    --output results/stats/caadrl_vs_heter_pdp40_clustered_greedy.csv
  "${PYTHON_BIN}" scripts/plot_route_comparison.py \
    --dataset "data/pdp/unified/pdp40_test_clustered_std${CLUSTER_STD}_seed${TEST_SEED}_n${TEST_SIZE}.pkl" \
    --caadrl-raw "${CAADRL_RAW}" \
    --heter-raw "${HETER_RAW}" \
    --instance-id 0 \
    --png figures/route_visualization_caadr_vs_heter.png \
    --pdf figures/route_visualization_caadr_vs_heter.pdf
else
  "${PYTHON_BIN}" scripts/compute_route_metrics.py \
    --dataset "data/pdp/unified/pdp40_test_clustered_std${CLUSTER_STD}_seed${TEST_SEED}_n${TEST_SIZE}.pkl" \
    --raw "${HETER_RAW}" \
    --output "${METRICS_DIR}/heter_pdp40_clustered_greedy.csv" \
    --method-label Heter \
    --variant-label baseline \
    --figure-instance 0 \
    --figure-png figures/route_visualization_heter.png \
    --figure-pdf figures/route_visualization_heter.pdf
  echo "TODO: matching CAADRL raw output is absent; paired improvement and CAADRL-vs-Heter route-comparison figure not generated."
fi
