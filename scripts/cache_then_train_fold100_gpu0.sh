#!/usr/bin/env bash
set -euo pipefail

cd /media/disk7t/FasterWAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/run_fold100_vae_cache_gpu0.sh
bash scripts/run_fold100_action_full_long_vae_cached_1gpu.sh
