#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_action_zero2.sh <nproc_per_node> [hydra_overrides...]}"
shift

MASTER_PORT="${MASTER_PORT:-29583}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_action.py \
  "$@"
