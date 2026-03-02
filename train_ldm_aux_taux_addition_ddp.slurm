#!/bin/bash
#SBATCH -J aux_ldm_brain_addition
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16                  # node count; override at submit if needed
#SBATCH --ntasks-per-node=1    # one launcher task per node
#SBATCH -t 24:00:00
#SBATCH -o /scratch/10102/hh29499/train_aux_ldm_wholebrain/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_aux_ldm_wholebrain/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/10102/hh29499/tracy/3D_Brain_partsLDM}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${PROJECT_ROOT}/train_ldm_aux_taux_addition_ddp.sh}"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "[slurm] launch script not found: ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

export CONFIG="${CONFIG:-configs/whole_brain_aux_taux_spacing1p5_ADDITION.json}"
export MAX_RESTARTS="${MAX_RESTARTS:-20}"
export MAX_REQUEUE="${MAX_REQUEUE:-10}"

if srun --export=ALL bash "${LAUNCH_SCRIPT}"; then
  rc=0
else
  rc=$?
fi

if [[ ${rc} -ne 0 && -n "${SLURM_JOB_ID:-}" && ${SLURM_RESTART_COUNT:-0} -lt ${MAX_REQUEUE} ]]; then
  echo "[slurm] requeueing job ${SLURM_JOB_ID} after rc=${rc} (restart_count=${SLURM_RESTART_COUNT:-0})"
  scontrol requeue "${SLURM_JOB_ID}"
fi

exit "${rc}"
