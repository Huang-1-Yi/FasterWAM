#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TASK_CONFIG=fold100_fasterwam_action_adapter
export RUN_ROOT=/media/sata4t/hy_fasterwam_runs/fold100
export RUN_ID="stage_a_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > "${RUN_ROOT}/latest_stage_a_run.txt"

echo "[stage-a] started=$(date --iso-8601=seconds) physical_gpu=${CUDA_VISIBLE_DEVICES} run_dir=${OUTPUT_DIR}"
bash scripts/train_action_single.sh 2>&1 | tee "${OUTPUT_DIR}/train.log"
