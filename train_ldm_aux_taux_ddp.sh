#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/whole_brain_aux_taux_spacing1p5.json}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_3d_brain_ldm_aux_taux_ddp.py}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Slurm-aware defaults.
NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

# Auto-detect visible GPU count when not specified.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_aux_taux_ddp_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] project_root=${PROJECT_ROOT}"
echo "[launch] config=${CONFIG}"
echo "[launch] nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK}"
echo "[launch] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
echo "[launch] log=${LOG_FILE}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config "${CONFIG}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

