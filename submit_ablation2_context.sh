#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/whole_brain_aux_taux_spacing1p5_CONTEXT.json}"
MASTER_PORT="${MASTER_PORT:-29620}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/train_ldm_aux_taux_context_ddp.sh}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

for f in "${CONFIG}" "${TRAIN_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[submit] required file not found: ${f}" >&2
    exit 1
  fi
done

echo "[run] project_root=${PROJECT_ROOT}"
echo "[run] config=${CONFIG} master_port=${MASTER_PORT} nproc_per_node=${NPROC_PER_NODE} max_restarts=${MAX_RESTARTS}"

CONFIG="${CONFIG}" \
MASTER_PORT="${MASTER_PORT}" \
MAX_RESTARTS="${MAX_RESTARTS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
bash "${TRAIN_SCRIPT}"
