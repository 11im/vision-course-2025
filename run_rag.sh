#!/bin/bash

# This script runs the RAG experiment on the full dataset.

set -e # Exit on error
export CUDA_LAUNCH_BLOCKING=1

# --- Configuration ---
PY="python3"
BASE="/work/donut-ocr-assisted"
JSON="${BASE}/data/docvqa/annotations/val_v1.0_withQT.json"
IMG_DIR="${BASE}/data/docvqa"
LOG_DIR="${BASE}/logs"
RESULTS_DIR="${BASE}/results"
TS=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

# --- RAG on GPU 0 ---
RAG_DIR="${RESULTS_DIR}/prompt_augmented_rag_paddle"
mkdir -p "${RAG_DIR}"
RAG_OUTPUT="${RAG_DIR}/rag_paddle_results.json"
RAG_LOG="${LOG_DIR}/rag_${TS}.log"

echo "Starting RAG experiment on GPU 0... Log: ${RAG_LOG}"
cd "${BASE}"

${PY} scripts/run_prompt_augmented_rag_paddle.py \
  --json_path "${JSON}" \
  --image_dir "${IMG_DIR}" \
  --output_file "${RAG_OUTPUT}" \
  --gpu 0 --ocr_gpu 0 \
  --top_k 5 --max_context_length 200 \
  2>&1 | tee "${RAG_LOG}"

echo "RAG experiment finished."
echo "Results: ${RAG_OUTPUT}"
echo "Log: ${RAG_LOG}"
