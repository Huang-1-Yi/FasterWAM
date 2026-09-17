#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false

export CACHE_DIR=/media/sata4t/hy_fasterwam_cache/fold100_vae_latents_wan22_batched4_v1
mkdir -p "${CACHE_DIR}"

python scripts/precompute_vae_latents.py \
  --task fold100_fasterwam_action_full_long \
  --cache-dir "${CACHE_DIR}" \
  --batch-size 4 \
  --num-workers 16 \
  --flush-every 128 \
  --log-every 100 \
  --splits train \
  2>&1 | tee "${CACHE_DIR}/precompute.log"
