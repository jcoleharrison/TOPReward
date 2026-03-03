#!/bin/bash
set -e

DATASET=$1

# Run Molmo 8B with chat template
echo "Running Molmo 8B on ${DATASET} with chat template"
uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_instruction_reward \
    dataset=$DATASET \
    model=molmo2_8b \
    model.model_name="allenai/Molmo2-8B" \
    prediction.num_examples=20 \
    prediction.add_chat_template=true \
    prediction.output_dir="${OUTPUT_DIR:-./results}"

# Run Molmo 4B with chat template
echo "Running Molmo 4B on ${DATASET} with chat template"
uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_instruction_reward \
    dataset=$DATASET \
    model=molmo2_4b \
    model.model_name="allenai/Molmo2-4B" \
    prediction.num_examples=20 \
    prediction.add_chat_template=true \
    prediction.output_dir="${OUTPUT_DIR:-./results}"
