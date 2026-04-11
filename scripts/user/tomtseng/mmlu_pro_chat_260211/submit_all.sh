#!/bin/bash
# Submit SLURM jobs to evaluate all 4 MMLU-Pro prompt format variants
# for each model-attack combination from the nov7_trial results.
#
# Usage: bash scripts/user/tomtseng/mmlu_pro_chat/submit_all.sh [--dry-run] [--skip-attack]

set -euo pipefail

DRY_RUN=false
SKIP_ATTACK=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-attack) SKIP_ATTACK=true; echo "=== SKIP-ATTACK MODE ===" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

REPO_DIR="/data/tom_tseng/TamperBench"
SOURCE_RESULTS_DIR="/data/saad_hossain/SafeTuneBed/results/nov7_trial"
OUTPUT_RESULTS_DIR="${REPO_DIR}/results/mmlu_pro_chat"
UV="/data/tom_tseng/.local/bin/uv"
SCRIPT="scripts/user/tomtseng/mmlu_pro_chat_260211/run_mmlu_pro_variants.py"

# Model alias -> HuggingFace model path mapping.
declare -A MODEL_PATHS=(
    # Llama 3.2
    ["llama3_1b_base"]="meta-llama/Llama-3.2-1B"
    ["llama3_1b_instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["llama3_3b_base"]="meta-llama/Llama-3.2-3B"
    ["llama3_3b_instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    # Llama 3 / 3.1 8B
    ["llama3_8b_base"]="meta-llama/Meta-Llama-3-8B"
    ["llama3_8b_instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    # Llama 3.1 8B defense variants
    ["llama3_8b_lat"]="sdhossain24/lat-llama3-8b-instruct-rt-jailbreak-robust1"
    ["llama3_8b_refat"]="samuelsimko/Meta-Llama-3-8B-Instruct-ReFAT"
    ["llama3_8b_rr"]="GraySwanAI/Llama-3-8B-Instruct-RR"
    ["llama3_8b_tar"]="lapisrocks/Llama-3-8B-Instruct-TAR-Refusal"
    ["llama3_8b_triplet_adv"]="samuelsimko/Meta-Llama-3-8B-Instruct-Triplet-Adv"
    # Mistral
    ["mistral_7b_base"]="mistralai/Mistral-7B-v0.1"
    ["mistral_7b_instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
    # Qwen3
    ["qwen3_0_6b"]="Qwen/Qwen3-0.6B"
    ["qwen3_0_6b_base"]="Qwen/Qwen3-0.6B-Base"
    ["qwen3_1_7b"]="Qwen/Qwen3-1.7B"
    ["qwen3_1_7b_base"]="Qwen/Qwen3-1.7B-Base"
    ["qwen3_4b"]="Qwen/Qwen3-4B"
    ["qwen3_4b_base"]="Qwen/Qwen3-4B-Base"
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["qwen3_8b_base"]="Qwen/Qwen3-8B-Base"
)

# Base model alias -> instruct model with chat template.
# Used to provide a chat template for base models that don't have one.
# Qwen3 base models already have a chat template
declare -A CHAT_TEMPLATE_MODELS=(
    ["llama3_1b_base"]="meta-llama/Llama-3.2-1B-Instruct"
    ["llama3_3b_base"]="meta-llama/Llama-3.2-3B-Instruct"
    ["llama3_8b_base"]="meta-llama/Llama-3.1-8B-Instruct"
    ["mistral_7b_base"]="mistralai/Mistral-7B-Instruct-v0.1"
)

# Attacks to evaluate (all standard attacks)
ATTACKS=(
    backdoor_finetune
    benign_full_parameter_finetune
    benign_lora_finetune
    competing_objectives_finetune
    embedding_attack
    full_parameter_finetune
    lora_finetune
    multilingual_finetune
    no_weight_modification
    style_modulation_finetune
)

submitted=0
skipped=0

for model_alias in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$model_alias]}"

    for attack in "${ATTACKS[@]}"; do
        best_json="${SOURCE_RESULTS_DIR}/${model_alias}/${attack}/optuna_single/best.json"

        # Skip if no best.json exists
        if [[ ! -f "$best_json" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        job_name="mmlu_chat_${model_alias}_${attack}"

        # Generate SLURM script
        slurm_script="#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=${OUTPUT_RESULTS_DIR}/logs/${model_alias}/${attack}_%j.out
#SBATCH --error=${OUTPUT_RESULTS_DIR}/logs/${model_alias}/${attack}_%j.err

cd ${REPO_DIR}

export WANDB_MODE=disabled
${UV} run ${SCRIPT} \"${model_path}\" \\
    --model-alias \"${model_alias}\" \\
    --attack \"${attack}\" \\
    --source-results-dir \"${SOURCE_RESULTS_DIR}\" \\
    --output-results-dir \"${OUTPUT_RESULTS_DIR}\""

        # Append --chat-template-model for base models
        if [[ -n "${CHAT_TEMPLATE_MODELS[$model_alias]+x}" ]]; then
            slurm_script+=" \\
    --chat-template-model \"${CHAT_TEMPLATE_MODELS[$model_alias]}\""
        fi

        if $SKIP_ATTACK; then
            slurm_script+=" --skip-attack"
        fi

        if $DRY_RUN; then
            echo "--- Would submit: ${model_alias}/${attack} ---"
            echo "$slurm_script"
            echo ""
        else
            # Create log directory
            mkdir -p "${OUTPUT_RESULTS_DIR}/logs/${model_alias}"
            echo "$slurm_script" | sbatch
            echo "Submitted: ${model_alias}/${attack}"
        fi

        submitted=$((submitted + 1))
    done
done

echo ""
echo "=== Summary ==="
echo "Jobs submitted (or would submit): ${submitted}"
echo "Skipped (no best.json): ${skipped}"
