#!/usr/bin/env bash
# Batch-run TOPReward predictions across multiple datasets using the last N frames.
#
# Usage:
#   ./scripts/batch_predict.sh [datasets_file]
#
# datasets_file defaults to scripts/datasets.txt (one HF dataset ID per line).
# Blank lines and lines starting with '#' are ignored.
#
# All Hydra overrides (model, num_frames, sampling, etc.) can be tweaked
# in the COMMON_OVERRIDES variable below.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_FILE="${1:-${SCRIPT_DIR}/batch_datasets.txt}"

if [[ ! -f "${DATASETS_FILE}" ]]; then
  echo "ERROR: datasets file not found: ${DATASETS_FILE}" >&2
  exit 1
fi

# Read datasets from file, skipping comments and blank lines
mapfile -t DATASETS < <(grep -v '^\s*#' "${DATASETS_FILE}" | grep -v '^\s*$')

FAILED_DATASETS=()

# Common overrides applied to every run
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

DS_COUNT=0
for ds in "${DATASETS[@]}"; do
  DS_COUNT=$((DS_COUNT + 1))

  # Clear HuggingFace / LeRobot cache every 5 datasets to free disk space
  if [ $((DS_COUNT % 1)) -eq 0 ] && [ "$DS_COUNT" -gt 0 ]; then
    echo "Clearing HuggingFace cache after ${DS_COUNT} datasets..."
    rm -rf "${HF_HOME:-${HOME}/.cache/huggingface}/lerobot"
    rm -rf "${HF_HOME:-${HOME}/.cache/huggingface}/batch_datasets"
  fi

  # Detect local path vs HuggingFace dataset ID
  DS_OVERRIDES=()
  if [[ "${ds}" == /* || "${ds}" == ./* ]]; then
    # Local path: extract repo_id from the last two path components (org/dataset)
    ds_repo_id="$(basename "$(dirname "${ds}")")/$(basename "${ds}")"
    DS_OVERRIDES+=("data_loader.root=${ds}" "dataset.dataset_name=${ds_repo_id}")
  else
    ds_repo_id="${ds}"
    DS_OVERRIDES+=("dataset.dataset_name=${ds}")
  fi

  # Derive a safe name for logging/output (replace / with _)
  ds_safe="${ds_repo_id//\//_}"
  echo "=========================================="
  echo "[${DS_COUNT}/${#DATASETS[@]}] Running TOPReward on dataset: ${ds}"
  echo "=========================================="
  if HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_topreward \
    "dataset.name=${ds_safe}" \
    "${DS_OVERRIDES[@]}" \
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
