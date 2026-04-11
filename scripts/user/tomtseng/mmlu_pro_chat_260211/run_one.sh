#!/bin/bash
# Run a single model-attack pair to verify the pipeline works.
#
# Usage:
#   bash scripts/user/tomtseng/mmlu_pro_chat_260211/run_one.sh <model_alias> <attack>
#
# Example:
#   bash scripts/user/tomtseng/mmlu_pro_chat_260211/run_one.sh qwen3_0_6b lora_finetune

set -euo pipefail

SKIP_ATTACK=false
args=()
for arg in "$@"; do
    case "$arg" in
        --skip-attack) SKIP_ATTACK=true ;;
        *) args+=("$arg") ;;
    esac
done

if [[ ${#args[@]} -lt 2 ]]; then
    echo "Usage: $0 <model_alias> <attack> [--skip-attack]"
    echo "Example: $0 qwen3_0_6b lora_finetune"
    exit 1
fi

MODEL_ALIAS="${args[0]}"
ATTACK="${args[1]}"

if [[ "$(hostname)" == "cais_login" ]]; then  # CAIS
    REPO_DIR="/data/tom_tseng/TamperBench"
    SOURCE_RESULTS_DIR="/data/saad_hossain/SafeTuneBed/results/nov7_trial"
    UV="/data/tom_tseng/.local/bin/uv"
else  # Flamingo
    REPO_DIR="/workspace/TamperBench"
    SOURCE_RESULTS_DIR="/tamperbench_data/cais_archive/nov7_trial"
    UV="uv"
fi
OUTPUT_RESULTS_DIR="${REPO_DIR}/results/mmlu_pro_chat"
SCRIPT="scripts/user/tomtseng/mmlu_pro_chat_260211/run_mmlu_pro_variants.py"

declare -A MODEL_PATHS=(
    ["llama3_1b_base"]="meta-llama/Llama-3.2-1B"
    ["llama3_1b_instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["llama3_3b_base"]="meta-llama/Llama-3.2-3B"
    ["llama3_3b_instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    ["llama3_8b_base"]="meta-llama/Meta-Llama-3-8B"
    ["llama3_8b_instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama3_8b_lat"]="sdhossain24/lat-llama3-8b-instruct-rt-jailbreak-robust1"
    ["llama3_8b_refat"]="samuelsimko/Meta-Llama-3-8B-Instruct-ReFAT"
    ["llama3_8b_rr"]="GraySwanAI/Llama-3-8B-Instruct-RR"
    ["llama3_8b_tar"]="lapisrocks/Llama-3-8B-Instruct-TAR-Refusal"
    ["llama3_8b_triplet_adv"]="samuelsimko/Meta-Llama-3-8B-Instruct-Triplet-Adv"
    ["mistral_7b_base"]="mistralai/Mistral-7B-v0.1"
    ["mistral_7b_instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
    ["qwen3_0_6b"]="Qwen/Qwen3-0.6B"
    ["qwen3_0_6b_base"]="Qwen/Qwen3-0.6B-Base"
    ["qwen3_1_7b"]="Qwen/Qwen3-1.7B"
    ["qwen3_1_7b_base"]="Qwen/Qwen3-1.7B-Base"
    ["qwen3_4b"]="Qwen/Qwen3-4B"
    ["qwen3_4b_base"]="Qwen/Qwen3-4B-Base"
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["qwen3_8b_base"]="Qwen/Qwen3-8B-Base"
)

declare -A CHAT_TEMPLATE_MODELS=(
    ["llama3_1b_base"]="meta-llama/Llama-3.2-1B-Instruct"
    ["llama3_3b_base"]="meta-llama/Llama-3.2-3B-Instruct"
    ["llama3_8b_base"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama3_8b_baseline"]="meta-llama/Llama-3.1-8B-Instruct"
    ["mistral_7b_base"]="mistralai/Mistral-7B-Instruct-v0.1"
    ["qwen3_0_6b_base"]="Qwen/Qwen3-0.6B"
    ["qwen3_1_7b_base"]="Qwen/Qwen3-1.7B"
    ["qwen3_4b_base"]="Qwen/Qwen3-4B"
    ["qwen3_8b_base"]="Qwen/Qwen3-8B"
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

echo "Running: ${MODEL_ALIAS} / ${ATTACK} (${MODEL_PATH})"
export WANDB_MODE=disabled

CMD=(
    "${UV}" run "${SCRIPT}" "${MODEL_PATH}"
    --model-alias "${MODEL_ALIAS}"
    --attack "${ATTACK}"
    --source-results-dir "${SOURCE_RESULTS_DIR}"
    --output-results-dir "${OUTPUT_RESULTS_DIR}"
)
if [[ -n "${CHAT_TEMPLATE_MODELS[$MODEL_ALIAS]+x}" ]]; then
    CMD+=(--chat-template-model "${CHAT_TEMPLATE_MODELS[$MODEL_ALIAS]}")
fi
if $SKIP_ATTACK; then
    CMD+=(--skip-attack)
fi
"${CMD[@]}"
