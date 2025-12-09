#!/bin/bash

# This script runs the Post-Correction experiment on the full dataset.

set -e # Exit on error

# --- Configuration ---
PY="python3"
BASE="/work/donut-ocr-assisted"
JSON="${BASE}/data/docvqa/annotations/val_v1.0_withQT.json"
IMG_DIR="${BASE}/data/docvqa"
LOG_DIR="${BASE}/logs"
RESULTS_DIR="${BASE}/results"
TS=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

# --- Post-Correction on GPU 0 ---
POST_DIR="${RESULTS_DIR}/post_correction_paddle"
mkdir -p "${POST_DIR}"
POST_OUTPUT="${POST_DIR}/post_correction_paddle_results.json"
POST_LOG="${LOG_DIR}/post_correction_${TS}.log"

echo "Starting Post-Correction experiment on GPU 0... Log: ${POST_LOG}"
cd "${BASE}"

${PY} scripts/run_post_correction_paddle.py \
  --json_path "${JSON}" \
  --image_dir "${IMG_DIR}" \
  --output_file "${POST_OUTPUT}" \
  --gpu 0 --ocr_gpu 0 \
  2>&1 | tee "${POST_LOG}"

echo "Post-Correction experiment finished."
echo "Results: ${POST_OUTPUT}"
echo "Log: ${POST_LOG}"
