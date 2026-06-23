#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIZES="${SIZES:-20 40 80}"
VARIANTS="${VARIANTS:-full no_enc_cluster cluster_only no_dec_cluster avg_fusion no_pomo}"
EPOCHS="${EPOCHS:-800}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"
TRAIN_EPISODES="${TRAIN_EPISODES:-2816}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_BATCH_FALLBACKS="${TRAIN_BATCH_FALLBACKS:-128 64 32 16 8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
TEST_SIZE="${TEST_SIZE:-10000}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
TEST_SEED="${TEST_SEED:-10000}"
CLUSTER_STD="${CLUSTER_STD:-0.1}"
TEST_OUT_DIR="${TEST_OUT_DIR:-data/pdp/unified}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/ablation}"
RAW_ROOT="${RAW_ROOT:-results/raw/ablation}"
SUMMARY_CSV="${SUMMARY_CSV:-results/summary/ablation_results.csv}"
SUMMARY_TEX="${SUMMARY_TEX:-results/summary/ablation_results.tex}"
FIGURE_PNG="${FIGURE_PNG:-figures/ablation_barplot.png}"
FIGURE_PDF="${FIGURE_PDF:-figures/ablation_barplot.pdf}"
LOG_ROOT="${LOG_ROOT:-logs/ablation}"

cd "${ROOT_DIR}"

mkdir -p "${LOG_ROOT}"

log_run() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "${log_path}")"
  {
    printf '\n[%s] ' "$(date -Iseconds)"
    printf '%q ' "$@"
    printf '\n'
  } | tee -a "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

checkpoint_epoch() {
  local checkpoint_name
  checkpoint_name="$(basename "$1")"
  checkpoint_name="${checkpoint_name#checkpoint-}"
  checkpoint_name="${checkpoint_name%.pt}"
  if [[ "${checkpoint_name}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${checkpoint_name}"
  else
    printf '%s\n' "-1"
  fi
}

latest_checkpoint_before_final() {
  local result_root="$1"
  find "${result_root}" -type f -name "checkpoint-*.pt" 2>/dev/null \
    | while read -r ckpt; do
        epoch="$(checkpoint_epoch "${ckpt}")"
        if [[ "${epoch}" -gt 0 && "${epoch}" -lt "${EPOCHS}" ]]; then
          printf '%08d %s\n' "${epoch}" "${ckpt}"
        fi
      done \
    | sort \
    | tail -n 1 \
    | cut -d' ' -f2-
}

train_with_resume() {
  local variant="$1"
  local size="$2"
  local result_root="$3"
  shift 3
  local batch_sizes=("$@")
  local batch_size

  for batch_size in "${batch_sizes[@]}"; do
    local checkpoint resume_checkpoint resume_epoch
    checkpoint="$(find "${result_root}" -type f -name "checkpoint-${EPOCHS}.pt" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${checkpoint}" ]]; then
      printf 'reuse existing %s\n' "${checkpoint}"
      return 0
    fi

    resume_checkpoint="$(latest_checkpoint_before_final "${result_root}" || true)"
    train_args=(
      "${PYTHON_BIN}" experiment.py
      --task train
      --ablation "${variant}"
      --problem_size "${size}"
      --distribution clustered
      --seed "${TRAIN_SEED}"
      --epochs "${EPOCHS}"
      --checkpoint-interval "${CHECKPOINT_INTERVAL}"
      --train_episodes "${TRAIN_EPISODES}"
      --train_batch_size "${batch_size}"
      --result_dir "${result_root}"
    )
    if [[ -n "${resume_checkpoint}" ]]; then
      resume_epoch="$(checkpoint_epoch "${resume_checkpoint}")"
      printf 'resume %s PDP%s from %s epoch %s with train batch %s\n' \
        "${variant}" "${size}" "${resume_checkpoint}" "${resume_epoch}" "${batch_size}"
      train_args+=(--resume_path "$(dirname "${resume_checkpoint}")" --resume_epoch "${resume_epoch}")
    else
      printf 'start %s PDP%s with train batch %s\n' "${variant}" "${size}" "${batch_size}"
    fi

    if log_run "${LOG_ROOT}/train_${variant}_pdp${size}.log" "${train_args[@]}"; then
      checkpoint="$(find "${result_root}" -type f -name "checkpoint-${EPOCHS}.pt" 2>/dev/null | sort | tail -n 1 || true)"
      if [[ -n "${checkpoint}" ]]; then
        return 0
      fi
    fi
    printf 'training attempt failed for %s PDP%s at batch %s; trying next fallback if available\n' \
      "${variant}" "${size}" "${batch_size}" | tee -a "${LOG_ROOT}/train_${variant}_pdp${size}.log"
  done

  printf 'missing checkpoint-%s.pt under %s after all train attempts\n' "${EPOCHS}" "${result_root}" >&2
  return 1
}

log_run "${LOG_ROOT}/build_unified_test_sets.log" \
  "${PYTHON_BIN}" scripts/build_unified_test_sets.py \
  --sizes ${SIZES} \
  --distributions clustered \
  --num-instances "${TEST_SIZE}" \
  --seed "${TEST_SEED}" \
  --cluster-std "${CLUSTER_STD}" \
  --name test \
  --out-dir "${TEST_OUT_DIR}"

for variant in ${VARIANTS}; do
  for size in ${SIZES}; do
    result_root="${ROOT_DIR}/${CHECKPOINT_ROOT}/pdp${size}_${variant}"
    checkpoint="$(find "${result_root}" -type f -name "checkpoint-${EPOCHS}.pt" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${checkpoint}" ]]; then
      train_with_resume "${variant}" "${size}" "${result_root}" "${TRAIN_BATCH_SIZE}" ${TRAIN_BATCH_FALLBACKS}
      checkpoint="$(find "${result_root}" -type f -name "checkpoint-${EPOCHS}.pt" | sort | tail -n 1)"
    fi

    dataset="${ROOT_DIR}/${TEST_OUT_DIR}/pdp${size}_test_clustered_std${CLUSTER_STD}_seed${TEST_SEED}_n${TEST_SIZE}.pkl"
    for decode in greedy sampling; do
      if [[ "${decode}" == "greedy" ]]; then
        width=0
        label="greedy"
      else
        width=1280
        label="sampling_1280"
      fi
      output="${ROOT_DIR}/${RAW_ROOT}/ablation_${variant}_pdp${size}_clustered_${label}.csv"
      if [[ -f "${output}" ]]; then
        echo "reuse existing ${output}"
        continue
      fi
      log_run "${LOG_ROOT}/eval_${variant}_pdp${size}_${label}.log" \
        "${PYTHON_BIN}" scripts/evaluate_caadrl.py \
        --dataset "${dataset}" \
        --checkpoint "${checkpoint}" \
        --output "${output}" \
        --ablation "${variant}" \
        --variant-label "${variant}" \
        --size "${size}" \
        --distribution clustered \
        --decode "${decode}" \
        --width "${width}" \
        --seed "${TEST_SEED}" \
        --num-instances "${TEST_SIZE}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --force
    done
  done
done

log_run "${LOG_ROOT}/summarize_ablation.log" \
  "${PYTHON_BIN}" scripts/summarize_raw_results.py \
  "${RAW_ROOT}/*.csv" \
  --csv-output "${SUMMARY_CSV}" \
  --tex-output "${SUMMARY_TEX}"

log_run "${LOG_ROOT}/plot_ablation.log" \
  "${PYTHON_BIN}" scripts/plot_ablation.py \
  --summary "${SUMMARY_CSV}" \
  --decode Greedy \
  --png "${FIGURE_PNG}" \
  --pdf "${FIGURE_PDF}"
