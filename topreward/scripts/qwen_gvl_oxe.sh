#!/bin/bash
set -e

DATASET=$1

uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_gvl \
    dataset=$DATASET \
    ++dataset.num_context_episodes=0 \
    model=qwen \
    model.model_name="Qwen/Qwen3-VL-8B-Instruct" \
    prediction.num_examples=20 \
    prediction.output_dir="${OUTPUT_DIR:-./results}"

# uv run python3 -m opengvl.scripts.predict \
#     --config-dir configs/experiments \
#     --config-name predict_gvl \
#     dataset=$DATASET \
#     ++dataset.num_context_episodes=0 \
#     model=qwen \
#     model.model_name="Qwen/Qwen3-VL-2B-Instruct" \
#     prediction.num_examples=20 \
#     prediction.output_dir="${OUTPUT_DIR:-./results}"

