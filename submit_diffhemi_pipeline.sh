#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

# Cluster sizing
NODES="${NODES:-12}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Retry/requeue controls propagated to the launch wrappers.
MAX_RESTARTS="${MAX_RESTARTS:-20}"
MAX_REQUEUE="${MAX_REQUEUE:-10}"

# Configs
LHEMI_AE_CONFIG="${LHEMI_AE_CONFIG:-configs/lhemi_AE_spacing1p5.json}"
LHEMI_LDM_CONFIG="${LHEMI_LDM_CONFIG:-configs/lhemi_LDM_spacing1p5.json}"
RHEMI_AE_CONFIG="${RHEMI_AE_CONFIG:-configs/rhemi_AE_spacing1p5.json}"
RHEMI_LDM_CONFIG="${RHEMI_LDM_CONFIG:-configs/rhemi_LDM_spacing1p5.json}"
FUSION_CONFIG="${FUSION_CONFIG:-configs/whole_brain_aux_taux_spacing1p5_DiffHEMI.json}"

# Distinct ports for concurrently running jobs.
MASTER_PORT_LHEMI_AE="${MASTER_PORT_LHEMI_AE:-29600}"
MASTER_PORT_LHEMI_LDM="${MASTER_PORT_LHEMI_LDM:-29601}"
MASTER_PORT_RHEMI_AE="${MASTER_PORT_RHEMI_AE:-29610}"
MASTER_PORT_RHEMI_LDM="${MASTER_PORT_RHEMI_LDM:-29611}"
MASTER_PORT_FUSION="${MASTER_PORT_FUSION:-29640}"

for f in \
  "${LHEMI_AE_CONFIG}" \
  "${LHEMI_LDM_CONFIG}" \
  "${RHEMI_AE_CONFIG}" \
  "${RHEMI_LDM_CONFIG}" \
  "${FUSION_CONFIG}" \
  "train_ae_lhemi.slurm" \
  "train_ldm_lhemi.slurm" \
  "train_ae_rhemi.slurm" \
  "train_ldm_rhemi.slurm" \
  "train_ldm_aux_taux_DiffHEMI_ddp.slurm"; do
  if [[ ! -f "${f}" ]]; then
    echo "[submit] required file not found: ${f}" >&2
    exit 1
  fi
done

submit_job() {
  local dep="$1"
  local config="$2"
  local port="$3"
  local slurm_script="$4"
  local jid

  if [[ -n "${dep}" ]]; then
    jid="$(sbatch --parsable \
      --dependency="afterok:${dep}" \
      -N "${NODES}" --ntasks-per-node="${NTASKS_PER_NODE}" --gpus-per-node="${GPUS_PER_NODE}" \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",NPROC_PER_NODE="${NPROC_PER_NODE}",CONFIG="${config}",MASTER_PORT="${port}",MAX_RESTARTS="${MAX_RESTARTS}",MAX_REQUEUE="${MAX_REQUEUE}" \
      "${slurm_script}")"
  else
    jid="$(sbatch --parsable \
      -N "${NODES}" --ntasks-per-node="${NTASKS_PER_NODE}" --gpus-per-node="${GPUS_PER_NODE}" \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",NPROC_PER_NODE="${NPROC_PER_NODE}",CONFIG="${config}",MASTER_PORT="${port}",MAX_RESTARTS="${MAX_RESTARTS}",MAX_REQUEUE="${MAX_REQUEUE}" \
      "${slurm_script}")"
  fi

  # Handle formats like "12345" and "12345;cluster".
  jid="${jid%%;*}"
  echo "${jid}"
}

echo "[submit] project_root=${PROJECT_ROOT}"
echo "[submit] resources: NODES=${NODES} NTASKS_PER_NODE=${NTASKS_PER_NODE} GPUS_PER_NODE=${GPUS_PER_NODE} NPROC_PER_NODE=${NPROC_PER_NODE}"

# 1) Each hemi pipeline: AE -> LDM
JID_LHEMI_AE="$(submit_job "" "${LHEMI_AE_CONFIG}" "${MASTER_PORT_LHEMI_AE}" "train_ae_lhemi.slurm")"
JID_LHEMI_LDM="$(submit_job "${JID_LHEMI_AE}" "${LHEMI_LDM_CONFIG}" "${MASTER_PORT_LHEMI_LDM}" "train_ldm_lhemi.slurm")"

JID_RHEMI_AE="$(submit_job "" "${RHEMI_AE_CONFIG}" "${MASTER_PORT_RHEMI_AE}" "train_ae_rhemi.slurm")"
JID_RHEMI_LDM="$(submit_job "${JID_RHEMI_AE}" "${RHEMI_LDM_CONFIG}" "${MASTER_PORT_RHEMI_LDM}" "train_ldm_rhemi.slurm")"

# 2) Fusion runs only after both hemi LDM jobs finish successfully.
JID_FUSION="$(submit_job "${JID_LHEMI_LDM}:${JID_RHEMI_LDM}" "${FUSION_CONFIG}" "${MASTER_PORT_FUSION}" "train_ldm_aux_taux_DiffHEMI_ddp.slurm")"

echo
echo "[submit] submitted jobs:"
echo "  lhemi_ae   : ${JID_LHEMI_AE}"
echo "  lhemi_ldm  : ${JID_LHEMI_LDM} (afterok:${JID_LHEMI_AE})"
echo "  rhemi_ae   : ${JID_RHEMI_AE}"
echo "  rhemi_ldm  : ${JID_RHEMI_LDM} (afterok:${JID_RHEMI_AE})"
echo "  fusion_ldm : ${JID_FUSION} (afterok:${JID_LHEMI_LDM}:${JID_RHEMI_LDM})"
echo
echo "[submit] inspect with: squeue -j ${JID_LHEMI_AE},${JID_LHEMI_LDM},${JID_RHEMI_AE},${JID_RHEMI_LDM},${JID_FUSION}"
