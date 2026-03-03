#!/bin/bash

# Run prediction experiments for all dataset and model combinations
# Each combination runs 3 times

set -e  # Exit on error

DATASET=$1

# Define models with config name and model_name override
# Format: "config_name:model_name_override"
models=(
    # "qwen:Qwen/Qwen3-VL-8B-Instruct"
    # "cosmos:nvidia/Cosmos-Reason1-7B"
    # "gemini:gemini-2.5-pro"
    "molmo2_8b:allenai/Molmo2-8B"
    # "molmo2_4b:allenai/Molmo2-4B"
)

# Number of runs per combination
num_runs=1

echo "Starting prediction experiments"
echo "Dataset: ${DATASET}"
echo "Models: ${#models[@]} model configurations"
echo "Runs per combination: $num_runs"
echo ""

total_experiments=$((${#models[@]} * num_runs))
current_experiment=0

for model_spec in "${models[@]}"; do
    # Split config_name:model_name
    IFS=':' read -r model_config model_name <<< "$model_spec"

    for run in $(seq 1 $num_runs); do
        current_experiment=$((current_experiment + 1))
        echo "================================================"
        echo "Experiment $current_experiment/$total_experiments"
        echo "Dataset: $DATASET | Model: $model_name | Run: $run"
        echo "================================================"

        HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m opengvl.scripts.predict \
            --config-dir configs/experiments \
            --config-name predict_gvl \
            dataset=$DATASET \
            ++dataset.num_context_episodes=0 \
            prediction.num_examples=20 \
            model=$model_config \
            model.model_name="$model_name" \
            prediction.output_dir="${OUTPUT_DIR:-./results}" \

        exit_code=$?
        # if [ $exit_code -ne 0 ]; then
        #     echo "ERROR: Experiment failed with exit code $exit_code"
        #     echo "Dataset: $dataset | Model: $model_name | Run: $run"
        #     # Continue to next experiment instead of exiting
        #     continue
        # fi

        echo "Completed run $run for $model_name on $DATASET"
        echo ""
    done
done

echo "================================================"
echo "All experiments completed!"
echo "================================================"
