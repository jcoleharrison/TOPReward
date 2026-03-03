#!/bin/bash
set -e

DATASET=$1

uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_instruction_reward \
    dataset=$DATASET \
    model=qwen \
    model.model_name="Qwen/Qwen3-VL-8B-Instruct" \
    prediction.num_examples=20 \
    prediction.add_chat_template=true \
    prediction.output_dir="${OUTPUT_DIR:-./results}"

uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_instruction_reward \
    dataset=$DATASET \
    model=qwen \
    model.model_name="Qwen/Qwen3-VL-2B-Instruct" \
    prediction.num_examples=20 \
    prediction.add_chat_template=true \
    prediction.output_dir="${OUTPUT_DIR:-./results}"
