#!/usr/bin/env bash
# Batch-run TOPReward predictions across multiple datasets using the last N frames.
#
# Usage:
#   ./scripts/batch_predict.sh
#
# Edit DATASETS below to list HuggingFace dataset IDs you want to run.
# No per-dataset config file needed — the default dataset config is used
# and dataset.dataset_name is overridden dynamically.
#
# All Hydra overrides (model, num_frames, sampling, etc.) can be tweaked
# in the COMMON_OVERRIDES variable below.

set -uo pipefail

FAILED_DATASETS=()

DATASETS=(
    RyanPan315464/stack_cube_so101
    ShishKebab04/toy-in-bowl
    badwolf256/so101_duck_picker
    ivlabs/so101_test
    k1000dai/so101_pick_sushi_from_shinkansen-pi05
    legalaspro/so101-greenblack-cube-cup-pnp-30hz
    legalaspro/so101-greenblack-cube-cup-pnp-50hz
    legalaspro/so101-pnp-crosslane-showcase-60-30hz-v0
    legalaspro/so101-pnp-crosslane-showcase-60-50hz-v0
    legalaspro/so101-pnp-microsanity-20-30hz-v0
    legalaspro/so101-pnp-microsanity-20-50hz-v0
    legalaspro/so101-ros-physical-ai-test
    lt-s/pick_one_block_v30
    marc-olivier-f/grab_cube_1
    marc-olivier-f/grab_cube_2
    marc-olivier-f/grab_cube_3
    miladherfeh/record-feb-10-python-2
    miladherfeh/record-feb12-final
    puheliang/lerobot_phl_grab_box_v2
    sreetz-nv/so101_teleop_vials_to_tray_camera_tweak
    sreetz-nv/so101_teleop_vials_to_tray_more_dr
    supreme-helix/act-ring-on-pole
    vvrs/so101-pick-place
)

# Common overrides applied to every run
COMMON_OVERRIDES=(
  sampling_method=last_n
  dataset.num_frames=3
  prediction.eval_all_episodes=true
  prediction.save_raw=true
)

for ds in "${DATASETS[@]}"; do
  # Derive a safe name for logging/output (replace / with _)
  ds_safe="${ds//\//_}"
  echo "=========================================="
  echo "Running TOPReward on dataset: ${ds}"
  echo "=========================================="
  if HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_topreward \
    "dataset.dataset_name=${ds}" \
    "dataset.name=${ds_safe}" \
    "${COMMON_OVERRIDES[@]}"; then
    echo "SUCCESS: ${ds}"
  else
    echo "FAILED: ${ds}"
    FAILED_DATASETS+=("${ds}")
  fi
done

echo ""
echo "=========================================="
echo "Batch complete. ${#FAILED_DATASETS[@]} dataset(s) failed."
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
  echo "Failed datasets:"
  for ds in "${FAILED_DATASETS[@]}"; do
    echo "  - ${ds}"
  done
fi
