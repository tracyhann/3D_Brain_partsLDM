#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# Usage:
#   bash train_ldm_aux_taux_addition_ddp.sh [config_json] [extra args...]
# Examples:
#   bash train_ldm_aux_taux_addition_ddp.sh
#   bash train_ldm_aux_taux_addition_ddp.sh configs/whole_brain_aux_taux_spacing1p5_ADDITION.json
#   NPROC_PER_NODE=8 bash train_ldm_aux_taux_addition_ddp.sh configs/whole_brain_aux_taux_spacing1p5_ADDITION.json --max_steps 40000

CONFIG_PATH="${CONFIG_PATH:-configs/whole_brain_aux_taux_spacing1p5_ADDITION.json}"
if [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
  shift
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  NNODES="${SLURM_NNODES:-1}"
  NODE_RANK="${SLURM_NODEID:-0}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  NNODES="${NNODES:-1}"
  NODE_RANK="${NODE_RANK:-0}"
fi

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_aux_taux_addition_ddp_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] project=${PROJECT_DIR}"
echo "[launch] config=${CONFIG_PATH}"
echo "[launch] nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launch] log=${LOG_FILE}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_3d_brain_ldm_aux_taux_addition_ddp.py \
  --config "${CONFIG_PATH}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

