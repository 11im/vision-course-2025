#!/bin/bash

# This script runs the Baseline experiment on the full dataset.

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

# --- Baseline on GPU 0 ---
BASELINE_DIR="${RESULTS_DIR}/baseline_fixed"
mkdir -p "${BASELINE_DIR}"
BASELINE_OUTPUT="${BASELINE_DIR}/baseline_fixed_results.json"
BASELINE_LOG="${LOG_DIR}/baseline_${TS}.log"

echo "Starting Baseline experiment on GPU 0... Log: ${BASELINE_LOG}"
cd "${BASE}"

${PY} scripts/run_baseline_fixed.py \
  --json_path "${JSON}" \
  --image_dir "${IMG_DIR}" \
  --output_file "${BASELINE_OUTPUT}" \
  --gpu 0 \
  2>&1 | tee "${BASELINE_LOG}"

echo "Baseline experiment finished."
echo "Results: ${BASELINE_OUTPUT}"
echo "Log: ${BASELINE_LOG}"
