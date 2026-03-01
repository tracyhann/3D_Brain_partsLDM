#!/bin/bash
#SBATCH -J aux_first
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16                # 节点数；按需改
#SBATCH --ntasks-per-node=1   # 每节点启动1个任务（每节点1 GPU）
#SBATCH -t 24:00:00
#SBATCH -o /scratch/10102/hh29499/train_ldm_aux_first/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_ldm_aux_first/%x_%j.err



srun --export=ALL \
  bash "/scratch/10102/hh29499/tracy/3D_Brain_partsLDM//train_ldm_aux_taux_ddp.sh"

