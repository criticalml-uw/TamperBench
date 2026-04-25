#!/usr/bin/env bash
# Run the best-SR attack configs for 6 selected models across 8 attack types + no_weight_modification.
# Total: 6 models × 9 configs = 54 runs.
#
# Usage:
#   bash scripts/whitebox/run_worst_sr_attacks.sh

set -euo pipefail

RUN_SCRIPT="/home/dev/SafeTuneBed/scripts/whitebox/run_single_attack.py"
CONFIGS_DIR="/home/dev/SafeTuneBed/configs/whitebox/attacks"
RESULTS_DIR="/home/dev/SafeTuneBed/results/worst_sr"
SEED=42

# Model HF paths and aliases
declare -A MODELS=(
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["llama3_8b_instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
    ["qwen3_4b"]="Qwen/Qwen3-4B"
    ["mistral_7b"]="mistralai/Mistral-7B-v0.1"
    ["qwen3_8b_base"]="Qwen/Qwen3-8B-Base"
    ["llama3_3b_instruct"]="meta-llama/Llama-3.2-3B-Instruct"
)

# Attack types with their config name suffix
# NOTE: benign_full_parameter_finetune and full_parameter_finetune are excluded
# because they OOM on a single 80GB GPU with 8B models (Adam optimizer states
# require ~64GB on top of model weights and gradients).
ATTACKS=(
    "no_weight_modification"
    "backdoor_finetune"
    "benign_full_parameter_finetune"
    "benign_lora_finetune"
    "competing_objectives_finetune"
    "full_parameter_finetune"
    "lora_finetune"
    "multilingual_finetune"
    "style_modulation_finetune"
)

for ALIAS in "${!MODELS[@]}"; do
    HF_PATH="${MODELS[$ALIAS]}"
    echo ""
    echo "============================================================"
    echo "  Model: ${ALIAS} (${HF_PATH})"
    echo "============================================================"

    for ATTACK in "${ATTACKS[@]}"; do
        # no_weight_modification uses "base" config; all others use "{alias}_worst_sr_a"
        if [[ "$ATTACK" == "no_weight_modification" ]]; then
            CONFIG_NAME="base"
        else
            CONFIG_NAME="${ALIAS}_worst_sr_a"
        fi

        echo ""
        echo "--- ${ALIAS} / ${ATTACK} / config=${CONFIG_NAME} ---"
        uv run "${RUN_SCRIPT}" \
            "${HF_PATH}" \
            --attack "${ATTACK}" \
            --config-name "${CONFIG_NAME}" \
            --configs-dir "${CONFIGS_DIR}" \
            --model-alias "${ALIAS}" \
            --results-dir "${RESULTS_DIR}" \
            --random-seed "${SEED}"
    done
done

echo ""
echo "All runs complete. Results in: ${RESULTS_DIR}"
