#!/bin/bash
# Submit SLURM jobs for the chat-attack regression test on CAIS cluster.
# Runs all 9 attacks (excluding embedding_attack) on qwen3_0_6b and qwen3_0_6b_base.
#
# Usage: bash scripts/user/tomtseng/chat_template_refactor_testing_260217/submit_all.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

REPO_DIR="/data/tom_tseng/TamperBench"
SOURCE_RESULTS_DIR="/data/saad_hossain/SafeTuneBed/results/nov7_trial"
OUTPUT_RESULTS_DIR="${REPO_DIR}/results/chat_template_refactor_testing_260217"
UV="/data/tom_tseng/.local/bin/uv"
SCRIPT="scripts/user/tomtseng/chat_template_refactor_testing_260217/run_regression.py"

declare -A MODEL_PATHS=(
    ["qwen3_0_6b"]="Qwen/Qwen3-0.6B"
    ["qwen3_0_6b_base"]="Qwen/Qwen3-0.6B-Base"
)

ATTACKS=(
    backdoor_finetune
    benign_full_parameter_finetune
    benign_lora_finetune
    competing_objectives_finetune
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

        if [[ ! -f "$best_json" ]]; then
            echo "Skipping ${model_alias}/${attack}: no best.json"
            skipped=$((skipped + 1))
            continue
        fi

        job_name="chat_reg_${model_alias}_${attack}"

        slurm_script="#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=${OUTPUT_RESULTS_DIR}/logs/${model_alias}/${attack}_%j.out
#SBATCH --error=${OUTPUT_RESULTS_DIR}/logs/${model_alias}/${attack}_%j.err

cd ${REPO_DIR}

export WANDB_MODE=disabled
${UV} run ${SCRIPT} \"${model_path}\" \\
    --model-alias \"${model_alias}\" \\
    --attack \"${attack}\" \\
    --source-results-dir \"${SOURCE_RESULTS_DIR}\" \\
    --output-results-dir \"${OUTPUT_RESULTS_DIR}\""

        if $DRY_RUN; then
            echo "--- Would submit: ${model_alias}/${attack} ---"
            echo "$slurm_script"
            echo ""
        else
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
