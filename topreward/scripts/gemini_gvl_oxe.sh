#!/bin/bash
set -e

DATASET=$1

uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_gvl \
    dataset=$DATASET \
    ++dataset.num_context_episodes=0 \
    model=gemini \
    model.model_name="gemini-2.5-pro" \
    prediction.output_dir="${OUTPUT_DIR:-./results}" \
    prediction.num_examples=20 \
