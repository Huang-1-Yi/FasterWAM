#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one explicitly approved physical GPU.}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "Expected one physical GPU, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29585}"

export TASK_CONFIG=fold100_fasterwam_action_full_long
export RUN_ROOT=/media/sata4t/hy_fasterwam_runs/fold100
export RUN_ID="action_full_long_1gpu_physical${CUDA_VISIBLE_DEVICES}_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > "${RUN_ROOT}/latest_action_full_long_1gpu_run.txt"

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "physical_cuda_device=${CUDA_VISIBLE_DEVICES}"
  echo "distributed=deepspeed_zero2_no_offload_single_process"
  echo "world_size=1"
  echo "master_port=${MASTER_PORT}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "task_config=${TASK_CONFIG}"
  echo "base_checkpoint=/media/sata4t/hy_fasterwam_checkpoints/fasterwam_release/robotwin/step_029355.pt"
  echo "action_delta_checkpoint=null"
  echo "max_steps=20000"
  echo "save_every=500"
  echo "eval_every=500"
  echo "gradient_accumulation_steps=8"
  echo "effective_batch_size=8"
  echo "python=$(python --version 2>&1)"
  python -c 'import torch; print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} | tee "${OUTPUT_DIR}/launch_manifest.txt"

echo "[action-full-long-1gpu] run_dir=${OUTPUT_DIR}"
bash scripts/train_action_zero2.sh 1 \
  "task=${TASK_CONFIG}" \
  "output_dir=${OUTPUT_DIR}" \
  "gradient_accumulation_steps=8" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
