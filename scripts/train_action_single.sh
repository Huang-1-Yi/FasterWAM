#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one verified available physical GPU first.}"

TASK_CONFIG="${TASK_CONFIG:-fold100_fasterwam_action_adapter}"
RUN_ROOT="${RUN_ROOT:-/media/sata4t/hy_fasterwam_runs/fold100}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/sata4t/hy_fasterwam_cache/hf_datasets}"
RUN_ID="${RUN_ID:-${TASK_CONFIG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}" "${HF_DATASETS_CACHE}"

echo "[launch] physical_cuda_devices=${CUDA_VISIBLE_DEVICES} task=${TASK_CONFIG} output=${OUTPUT_DIR} hf_datasets_cache=${HF_DATASETS_CACHE}"
exec python -u scripts/train_action.py \
  "task=${TASK_CONFIG}" \
  "output_dir=${OUTPUT_DIR}" \
  "$@"
