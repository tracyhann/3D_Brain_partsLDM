#!/bin/bash
#SBATCH -J submit_no_shared_ae
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/10102/hh29499/submit_no_shared_ae_pipeline/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/submit_no_shared_ae_pipeline/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/10102/hh29499/tracy/3D_Brain_partsLDM}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${PROJECT_ROOT}/submit_ldm_no_shared_ae_pipeline.sh}"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "[slurm] launch script not found: ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

export NODES="${NODES:-16}"
export NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export LHEMI_LDM_CONFIG="${LHEMI_LDM_CONFIG:-configs/lhemi_LDM_spacing1p5.json}"
export RHEMI_LDM_CONFIG="${RHEMI_LDM_CONFIG:-configs/rhemi_LDM_spacing1p5.json}"
export FUSION_CONFIG="${FUSION_CONFIG:-configs/whole_brain_aux_taux_spacing1p5_DiffHEMI.json}"
export MASTER_PORT_LHEMI_LDM="${MASTER_PORT_LHEMI_LDM:-29600}"
export MASTER_PORT_RHEMI_LDM="${MASTER_PORT_RHEMI_LDM:-29610}"
export MASTER_PORT_FUSION="${MASTER_PORT_FUSION:-29640}"

srun --export=ALL bash "${LAUNCH_SCRIPT}"
