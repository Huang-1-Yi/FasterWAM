#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU, for example 0.}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29586}"

export TASK_CONFIG=fold100_fasterwam_action_full_long_vae_cached
export RUN_ROOT=/media/sata4t/hy_fasterwam_runs/fold100
export RUN_ID="action_full_long_vae_cached_1gpu_physical${CUDA_VISIBLE_DEVICES}_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > "${RUN_ROOT}/latest_action_full_long_vae_cached_1gpu_run.txt"

{
  echo "started=$(date --iso-8601=seconds)"
  echo "physical_cuda_device=${CUDA_VISIBLE_DEVICES}"
  echo "task_config=${TASK_CONFIG}"
  echo "vae_latent_cache=/media/sata4t/hy_fasterwam_cache/fold100_vae_latents_wan22_batched4_v1"
  echo "base_checkpoint=/media/sata4t/hy_fasterwam_checkpoints/fasterwam_release/robotwin/step_029355.pt"
  echo "max_steps=20000"
  echo "gradient_accumulation_steps=8"
  echo "effective_batch_size=8"
  echo "git_commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
} | tee "${OUTPUT_DIR}/launch_manifest.txt"

bash scripts/train_action_zero2.sh 1 \
  "task=${TASK_CONFIG}" \
  "output_dir=${OUTPUT_DIR}" \
  "gradient_accumulation_steps=8" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
