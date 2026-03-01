#!/bin/bash
#SBATCH -J aux_taux_add
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -t 40:00:00
#SBATCH -o /scratch/10102/hh29499/train_ldm_aux_taux_add/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_ldm_aux_taux_add/%x_%j.err

set -euo pipefail

# Optional overrides at submit time:
#   sbatch --export=ALL,CONFIG_PATH=configs/whole_brain_aux_taux_spacing1p5_ADDITION.json,NPROC_PER_NODE=8 train_ldm_aux_taux_addition_ddp.slurm

PROJECT_DIR="/ssd2/wenyan/CarTwin/3dbrain/second/3D_Brain_partsLDM"
mkdir -p /scratch/10102/hh29499/train_ldm_aux_taux_add

srun --export=ALL \
  bash "${PROJECT_DIR}/train_ldm_aux_taux_addition_ddp.sh"

