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
cd /scratch/10102/hh29499/tracy/3D_Brain_partsLDM

CONFIG_PATH="${CONFIG_PATH:-configs/whole_brain_aux_taux_spacing1p5_ADDITION.json}"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500   # 任意空闲端口
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_aux_taux_addition_ddp_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"


torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_3d_brain_ldm_aux_taux_addition_ddp.py \
  --config "${CONFIG_PATH}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

