#!/bin/bash

# This script runs the Random-Sampling experiment on the full dataset.

set -e # Exit on error
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# --- Configuration ---
PY="python3"
BASE="/work/donut-ocr-assisted"
JSON="${BASE}/data/docvqa/annotations/val_v1.0_withQT.json"
IMG_DIR="${BASE}/data/docvqa"
LOG_DIR="${BASE}/logs"
RESULTS_DIR="${BASE}/results"
TS=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

# --- Random-Sampling on GPU 1 ---
RANDOM_DIR="${RESULTS_DIR}/prompt_augmented_random_paddle"
mkdir -p "${RANDOM_DIR}"
RANDOM_OUTPUT="${RANDOM_DIR}/random_paddle_results.json"
RANDOM_LOG="${LOG_DIR}/random_${TS}.log"

echo "Starting Random-Sampling experiment on GPU 1... Log: ${RANDOM_LOG}"
cd "${BASE}"

${PY} scripts/run_prompt_augmented_random_paddle.py \
  --json_path "${JSON}" \
  --image_dir "${IMG_DIR}" \
  --output_file "${RANDOM_OUTPUT}" \
  --gpu 0 --ocr_gpu 0 \
  --start_idx 2653 \
  --num_segments 5 --max_context_length 200 \

echo "Random-Sampling experiment finished."
echo "Results: ${RANDOM_OUTPUT}"
echo "Log: ${RANDOM_LOG}"
