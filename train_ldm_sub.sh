#!/usr/bin/env bash
set -euo pipefail

cd /scratch/10102/hh29499/tracy/3D_Brain_partsLDM

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500   # 任意空闲端口
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID

LOG_DIR="/scratch/10102/hh29499/tracy/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"


torchrun \
  --nnodes=$NNODES \
  --nproc_per_node 1 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/train_3d_brain_ldm_steps_ddp.py \
  --config configs/sub_LDM_spacing1p5.json \
  2>&1 | tee "$LOG"
