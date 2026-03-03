#!/bin/bash
set -e

DATASET=$1

uv run python3 -m opengvl.scripts.predict \
    --config-dir configs/experiments \
    --config-name predict_instruction_reward \
    dataset=$DATASET \
    model=gemini_ir \
    prediction.num_examples=20 \
    prediction.output_dir="${OUTPUT_DIR:-./results}"
