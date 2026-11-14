#!/bin/bash
# Run a single model-attack pair for the chat-attack regression test.
#
# Usage:
#   bash scripts/user/tomtseng/chat_template_refactor_testing_260217/run_one.sh <model_alias> <attack>
#
# Example:
#   bash scripts/user/tomtseng/chat_template_refactor_testing_260217/run_one.sh qwen3_0_6b lora_finetune

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_alias> <attack>"
    echo "Example: $0 qwen3_0_6b lora_finetune"
    exit 1
fi

MODEL_ALIAS="$1"
ATTACK="$2"

if [[ "$(hostname)" == "cais_login" ]]; then
    REPO_DIR="/data/tom_tseng/TamperBench"
    SOURCE_RESULTS_DIR="/data/saad_hossain/SafeTuneBed/results/nov7_trial"
    UV="/data/tom_tseng/.local/bin/uv"
else
    REPO_DIR="/workspace/TamperBench"
    SOURCE_RESULTS_DIR="/tamperbench_data/cais_archive/nov7_trial"
    UV="uv"
fi
OUTPUT_RESULTS_DIR="${REPO_DIR}/results/chat_template_refactor_testing_260217"
SCRIPT="scripts/user/tomtseng/chat_template_refactor_testing_260217/run_regression.py"

declare -A MODEL_PATHS=(
    ["qwen3_0_6b"]="Qwen/Qwen3-0.6B"
    ["qwen3_0_6b_base"]="Qwen/Qwen3-0.6B-Base"
)

if [[ -z "${MODEL_PATHS[$MODEL_ALIAS]+x}" ]]; then
    echo "Error: Unknown model alias '${MODEL_ALIAS}'"
    echo "Available: ${!MODEL_PATHS[*]}"
    exit 1
fi

MODEL_PATH="${MODEL_PATHS[$MODEL_ALIAS]}"

BEST_JSON="${SOURCE_RESULTS_DIR}/${MODEL_ALIAS}/${ATTACK}/optuna_single/best.json"
if [[ ! -f "$BEST_JSON" ]]; then
    echo "Error: No best.json at ${BEST_JSON}"
    exit 1
fi

cd "${REPO_DIR}"

echo "Running regression test: ${MODEL_ALIAS} / ${ATTACK} (${MODEL_PATH})"
export WANDB_MODE=disabled

"${UV}" run "${SCRIPT}" "${MODEL_PATH}" \
    --model-alias "${MODEL_ALIAS}" \
    --attack "${ATTACK}" \
    --source-results-dir "${SOURCE_RESULTS_DIR}" \
    --output-results-dir "${OUTPUT_RESULTS_DIR}"
