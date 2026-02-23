#!/bin/bash
#SBATCH -J sub
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 40:00:00
#SBATCH -o /scratch/10102/hh29499/train_ldm_sub/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_ldm_sub/%x_%j.err

srun --export=ALL \
  bash /scratch/10102/hh29499/tracy/3D_Brain_partsLDM/train_ldm_sub.sh
