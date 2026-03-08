#!/bin/bash
#SBATCH -J mask_ctx_seq
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16                  # node count; override at submit if needed
#SBATCH --ntasks-per-node=1    # one launcher task per node
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/10102/hh29499/train_ldm_mask_context_sequence/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_ldm_mask_context_sequence/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/10102/hh29499/tracy/3D_Brain_partsLDM}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${PROJECT_ROOT}/train_ldm_mask_context_sequence_ddp.sh}"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "[slurm] launch script not found: ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

export STEP1_CONFIG="${STEP1_CONFIG:-configs/whole_brain_maskLDM_spacing1p5.json}"
export STEP2_CONFIG="${STEP2_CONFIG:-configs/whole_brain_aux_taux_spacing1p5_CONTEXT.json}"
export STEP1_MASTER_PORT="${STEP1_MASTER_PORT:-29610}"
export STEP2_MASTER_PORT="${STEP2_MASTER_PORT:-29620}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if srun --export=ALL bash "${LAUNCH_SCRIPT}"; then
  rc=0
else
  rc=$?
fi

exit "${rc}"
