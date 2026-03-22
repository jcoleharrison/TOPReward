#!/bin/bash
# Launch TOPReward batch predictions on Beaker via gantry.
#
# Usage:
#   ./scripts/beaker_batch_predict.sh [datasets_file]
#
# datasets_file defaults to scripts/datasets.txt (one HF dataset ID per line).
# Blank lines and lines starting with '#' are ignored.
#
# Each dataset gets its own Beaker job (launched in the background with &).
# Edit COMMON_OVERRIDES and the gantry flags below as needed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_FILE="${1:-${SCRIPT_DIR}/batch_datasets.txt}"

if [[ ! -f "${DATASETS_FILE}" ]]; then
  echo "ERROR: datasets file not found: ${DATASETS_FILE}" >&2
  exit 1
fi

# Read datasets from file, skipping comments and blank lines
mapfile -t DATASETS < <(grep -v '^\s*#' "${DATASETS_FILE}" | grep -v '^\s*$')

# ---------------------------------------------------------------------------
# Common Hydra overrides applied to every run
# ---------------------------------------------------------------------------
COMMON_OVERRIDES=(
  sampling_method=last_n
  dataset.num_frames=3
  prediction.eval_all_episodes=false
  prediction.num_examples=10
  dataset.max_episodes=10
  dataset.num_context_episodes=0
  prediction.save_raw=true
  prediction.continue_on_error=true
  prediction.predict_last_n_prefixes=3
)

# ---------------------------------------------------------------------------
# Beaker / Gantry settings — edit these to match your environment
# ---------------------------------------------------------------------------
BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
CLUSTER="ai2/ceres"
WORKSPACE="ai2/molmo-act"
PRIORITY="urgent"
WEKA_MOUNT="oe-training-default:/mount/weka"
OUTPUT_BASE="/mount/weka/shiruic/topreward/results"

# ---------------------------------------------------------------------------
# Launch one Beaker job per dataset
# ---------------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
  # Detect local path vs HuggingFace dataset ID
  DS_OVERRIDE_ARGS=""
  if [[ "${ds}" == /* || "${ds}" == ./* ]]; then
    ds_repo_id="$(basename "$(dirname "${ds}")")/$(basename "${ds}")"
    DS_OVERRIDE_ARGS="data_loader.root=${ds} dataset.dataset_name=${ds_repo_id}"
  else
    ds_repo_id="${ds}"
    DS_OVERRIDE_ARGS="dataset.dataset_name=${ds}"
  fi

  # Derive a safe name (replace / with _)
  ds_safe="${ds_repo_id//\//_}"

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="TOPReward_${ds_safe}_${TIMESTAMP}"
  # Beaker name max 128 chars
  BEAKER_NAME="${RUN_NAME:0:123}"

  # Build the override string for this dataset
  OVERRIDE_ARGS=""
  for ov in "${COMMON_OVERRIDES[@]}"; do
    OVERRIDE_ARGS="${OVERRIDE_ARGS} ${ov}"
  done

  echo "Launching: ${BEAKER_NAME}"

  gantry run \
    --allow-dirty \
    --beaker-image "${BEAKER_IMAGE}" \
    --gpus=1 \
    --cluster="${CLUSTER}" \
    --workspace="${WORKSPACE}" \
    --weka="${WEKA_MOUNT}" \
    --name="${BEAKER_NAME}" \
    --task-name="${BEAKER_NAME}" \
    --description="${RUN_NAME}" \
    --priority="${PRIORITY}" \
    --install="uv pip install -e ." \
    --env OUTPUT_DIR="${OUTPUT_BASE}/${ds_safe}" \
    --env-secret HF_TOKEN=hf_token_shirui \
    -- \
    bash -c "HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
      --config-dir configs/experiments \
      --config-name predict_topreward \
      ${DS_OVERRIDE_ARGS} \
      dataset.name=${ds_safe} \
      prediction.output_dir=\${OUTPUT_DIR:-./results} \
      ${OVERRIDE_ARGS}" &
done

echo ""
echo "All jobs submitted. Use 'beaker session list' or the Beaker UI to monitor progress."
