#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_eval_all_samples.sh
# Optional:
#   PROJECT_ROOT=/DATA2/lulin2/tracy/3D_Brain_partsLDM bash scripts/run_eval_all_samples.sh

PROJECT_ROOT="${PROJECT_ROOT:-/DATA2/lulin2/tracy/3D_Brain_partsLDM}"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/evaluation.py"
CSV_PATH="$PROJECT_ROOT/data/processed_parts/whole_brain+3parts+masks_0206.csv"
SPLIT_JSON="$PROJECT_ROOT/data/patient_splits_image_ids_75_10_15.json"
MEDICALNET_CKPT_DIR="$PROJECT_ROOT/ckpts/medicalnet"

SAMPLE_DIRS=(
  "$PROJECT_ROOT/samples/baseline_LDM"
  "$PROJECT_ROOT/samples/real_test_samples"
  "$PROJECT_ROOT/samples/whole_brain_aux_taux_NO_AUX_UNET_spacing1p5"
  "$PROJECT_ROOT/samples/whole_brain_aux_taux_NO_INJ_UNET_spacing1p5"
  "$PROJECT_ROOT/samples/whole_brain_aux_taux_UNET_spacing1p5"
  "$PROJECT_ROOT/samples/whole_brain_mask_UNET_spacing1p5_ddp"
)

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: evaluation script not found: $EVAL_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: csv not found: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "ERROR: split json not found: $SPLIT_JSON" >&2
  exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Evaluation script: $EVAL_SCRIPT"
echo "CSV: $CSV_PATH"
echo "Split JSON: $SPLIT_JSON"
echo

for GENERATED_DIR in "${SAMPLE_DIRS[@]}"; do
  if [[ ! -d "$GENERATED_DIR" ]]; then
    echo "Skipping missing directory: $GENERATED_DIR"
    continue
  fi

  echo "============================================================"
  echo "Evaluating: $GENERATED_DIR"
  python3 "$EVAL_SCRIPT" \
    --generated_dir "$GENERATED_DIR" \
    --csv "$CSV_PATH" \
    --data_split_json_path "$SPLIT_JSON" \
    --image_key whole_brain \
    --dist_feature_mode inception2d \
    --fid_slice_axes ax,cor,sag \
    --fid_slices_per_axis 16 \
    --fid_slice_margin 16 \
    --mmd_kernel rbf \
    --intensity_mode normalize_both \
    --norm_percentiles 0.5,99.5 \
    --use_medicalnet_mmd3d \
    --medicalnet_model medicalnet_resnet10_23datasets \
    --medicalnet_ckpt_dir "$MEDICALNET_CKPT_DIR"
done

echo
echo "All evaluations completed."
