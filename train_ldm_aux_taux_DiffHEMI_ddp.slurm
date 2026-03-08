#!/bin/bash
#SBATCH -J aux_diffhemi
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 40:00:00
#SBATCH -o /scratch/10102/hh29499/train_ldm_aux_diffhemi/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_ldm_aux_diffhemi/%x_%j.err

srun --export=ALL \
  bash /scratch/10102/hh29499/tracy/3D_Brain_partsLDM/train_ldm_aux_taux_DiffHEMI_ddp.sh
