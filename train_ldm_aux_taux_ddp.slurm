#!/bin/bash
#SBATCH -J aux_taux_ddp
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH -t 40:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# Optional overrides at submit time:
#   sbatch --export=ALL,CONFIG=configs/xxx.json,NPROC_PER_NODE=8 train_ldm_aux_taux_ddp.slurm
export CONFIG="${CONFIG:-configs/whole_brain_aux_taux_spacing1p5.json}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29500}"

srun --export=ALL \
  bash "${PROJECT_ROOT}/train_ldm_aux_taux_ddp.sh"

