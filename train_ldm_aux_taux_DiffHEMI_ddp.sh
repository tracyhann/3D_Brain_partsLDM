#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-${CONFIG_PATH:-configs/whole_brain_aux_taux_spacing1p5_DiffHEMI.json}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_3d_brain_ldm_aux_taux_DiffHEMI_ddp.py}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_SLEEP_SEC="${RESTART_SLEEP_SEC:-15}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[launch] config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[launch] training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

# Slurm-aware defaults.
NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

# Auto-detect visible GPU count when not specified.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

# Prefer config-defined run naming; env vars can override.
CFG_OUTDIR="$(python -c 'import json,sys; d=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(d.get("outdir",""))' "${CONFIG}" 2>/dev/null || true)"
CFG_PREFIX="$(python -c 'import json,sys; d=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(d.get("out_prefix",""))' "${CONFIG}" 2>/dev/null || true)"
CFG_POSTFIX="$(python -c 'import json,sys; d=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(d.get("out_postfix",""))' "${CONFIG}" 2>/dev/null || true)"

OUTDIR_BASE="${OUTDIR_BASE:-${CFG_OUTDIR:-ckpts/UNET}}"
OUT_PREFIX="${OUT_PREFIX:-${CFG_PREFIX:-whole_brain_aux_taux_DiffHEMI_UNET}}"
OUT_POSTFIX="${OUT_POSTFIX:-${CFG_POSTFIX:-resume}}"

if [[ "${OUTDIR_BASE}" = /* ]]; then
  OUTDIR_ABS="${OUTDIR_BASE}"
else
  OUTDIR_ABS="${PROJECT_ROOT}/${OUTDIR_BASE}"
fi

if [[ -n "${OUT_PREFIX}" ]]; then
  RUN_NAME="${OUT_PREFIX}_${OUT_POSTFIX}"
else
  RUN_NAME="${OUT_POSTFIX}"
fi
RUN_DIR="${OUTDIR_ABS}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_aux_taux_diffhemi_ddp_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] project_root=${PROJECT_ROOT}"
echo "[launch] config=${CONFIG}"
echo "[launch] train_script=${TRAIN_SCRIPT}"
echo "[launch] nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK}"
echo "[launch] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] log=${LOG_FILE}"

attempt=0
while (( attempt <= MAX_RESTARTS )); do
  RESUME_ARGS=()
  if [[ -f "${RUN_DIR}/UNET_last.pt" ]]; then
    RESUME_ARGS=(--resume_ckpt "${RUN_DIR}/UNET_last.pt")
  fi

  if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "[launch] attempt=${attempt}/${MAX_RESTARTS} resume_from=${RUN_DIR}/UNET_last.pt"
  else
    echo "[launch] attempt=${attempt}/${MAX_RESTARTS} fresh_start"
  fi

  set +e
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    --outdir "${OUTDIR_BASE}" \
    --out_prefix "${OUT_PREFIX}" \
    --out_postfix "${OUT_POSTFIX}" \
    "${RESUME_ARGS[@]}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[launch] training finished successfully"
    exit 0
  fi

  if (( attempt >= MAX_RESTARTS )); then
    echo "[launch] failed with rc=${rc}; reached max restarts (${MAX_RESTARTS})" >&2
    exit "${rc}"
  fi

  if [[ ! -f "${RUN_DIR}/UNET_last.pt" ]]; then
    echo "[launch] failed with rc=${rc}; no checkpoint found at ${RUN_DIR}/UNET_last.pt" >&2
    exit "${rc}"
  fi

  attempt=$((attempt + 1))
  echo "[launch] failed with rc=${rc}; sleeping ${RESTART_SLEEP_SEC}s then retrying from checkpoint"
  sleep "${RESTART_SLEEP_SEC}"
done
