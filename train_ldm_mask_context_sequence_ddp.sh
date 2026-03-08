#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

STEP1_LAUNCH_SCRIPT="${STEP1_LAUNCH_SCRIPT:-${PROJECT_ROOT}/train_ldm_mask_ddp.sh}"
STEP2_LAUNCH_SCRIPT="${STEP2_LAUNCH_SCRIPT:-${PROJECT_ROOT}/train_ldm_aux_taux_context_ddp.sh}"

STEP1_CONFIG="${STEP1_CONFIG:-configs/whole_brain_maskLDM_spacing1p5.json}"
STEP2_CONFIG="${STEP2_CONFIG:-configs/whole_brain_aux_taux_spacing1p5_CONTEXT.json}"
STEP1_MASTER_PORT="${STEP1_MASTER_PORT:-29610}"
STEP2_MASTER_PORT="${STEP2_MASTER_PORT:-29620}"

if [[ ! -f "${STEP1_LAUNCH_SCRIPT}" ]]; then
  echo "[pipeline] step1 launch script not found: ${STEP1_LAUNCH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${STEP2_LAUNCH_SCRIPT}" ]]; then
  echo "[pipeline] step2 launch script not found: ${STEP2_LAUNCH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${STEP1_CONFIG}" ]]; then
  echo "[pipeline] step1 config not found: ${STEP1_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${STEP2_CONFIG}" ]]; then
  echo "[pipeline] step2 config not found: ${STEP2_CONFIG}" >&2
  exit 1
fi

echo "[pipeline] project_root=${PROJECT_ROOT}"
echo "[pipeline] step1(mask): config=${STEP1_CONFIG} master_port=${STEP1_MASTER_PORT}"
echo "[pipeline] step2(context): config=${STEP2_CONFIG} master_port=${STEP2_MASTER_PORT}"

echo "[pipeline] starting step1: mask LDM"
if CONFIG="${STEP1_CONFIG}" MASTER_PORT="${STEP1_MASTER_PORT}" \
  bash "${STEP1_LAUNCH_SCRIPT}" "$@"; then
  echo "[pipeline] step1 finished successfully"
else
  rc=$?
  echo "[pipeline] step1 failed with rc=${rc}" >&2
  exit "${rc}"
fi

echo "[pipeline] starting step2: context cLDM"
if CONFIG="${STEP2_CONFIG}" MASTER_PORT="${STEP2_MASTER_PORT}" \
  bash "${STEP2_LAUNCH_SCRIPT}" "$@"; then
  echo "[pipeline] step2 finished successfully"
else
  rc=$?
  echo "[pipeline] step2 failed with rc=${rc}" >&2
  exit "${rc}"
fi

echo "[pipeline] all steps finished successfully"
