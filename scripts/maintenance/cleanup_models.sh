#!/bin/bash
# Continuously check completion and delete cached models every hour

cd /data/saad_hossain/SafeTuneBed

SEEDS=(22 42 102)
HF_CACHE="/data/saad_hossain/cache/huggingface/hub"
DRY_RUN=false

# Parse command line arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Model name mappings (alias -> HF cache name)
declare -A MODEL_CACHE_NAMES=(
    ["qwen3_0_6b_base"]="models--Qwen--Qwen3-0.6B-Base"
    ["qwen3_0_6b"]="models--Qwen--Qwen3-0.6B"
    ["llama3_1b_base"]="models--meta-llama--Llama-3.2-1B"
    ["llama3_1b_instruct"]="models--meta-llama--Llama-3.2-1B-Instruct"
    ["qwen3_1_7b_base"]="models--Qwen--Qwen3-1.7B-Base"
    ["qwen3_1_7b"]="models--Qwen--Qwen3-1.7B"
    ["llama3_3b_base"]="models--meta-llama--Llama-3.2-3B"
    ["llama3_3b_instruct"]="models--meta-llama--Llama-3.2-3B-Instruct"
    ["qwen3_4b_base"]="models--Qwen--Qwen3-4B-Base"
    ["qwen3_4b"]="models--Qwen--Qwen3-4B"
    ["mistral_7b_base"]="models--mistralai--Mistral-7B-v0.1"
    ["mistral_7b_instruct"]="models--mistralai--Mistral-7B-Instruct-v0.1"
    ["qwen3_8b_base"]="models--Qwen--Qwen3-8B-Base"
    ["qwen3_8b"]="models--Qwen--Qwen3-8B"
    ["llama3_8b_baseline"]="models--meta-llama--Meta-Llama-3-8B"
    ["llama3_8b_instruct_baseline"]="models--meta-llama--Meta-Llama-3-8B-Instruct"
    ["llama3_8b_rr"]="models--GraySwanAI--Llama-3-8B-Instruct-RR"
    ["llama3_8b_refat"]="models--samuelsimko--Meta-Llama-3-8B-Instruct-ReFAT"
    ["llama3_8b_triplet_adv"]="models--samuelsimko--Meta-Llama-3-8B-Instruct-Triplet-Adv"
    ["llama3_8b_lat"]="models--sdhossain24--lat-llama3-8b-instruct-rt-jailbreak-robust1"
    ["llama3_8b_tar"]="models--lapisrocks--Llama-3-8B-Instruct-TAR-Refusal"
)

ATTACKS=(
    "no_weight_modification"
    "backdoor_finetune"
    "multilingual_finetune"
    "competing_objectives_finetune"
    "full_parameter_finetune"
    "style_modulation_finetune"
    "benign_full_parameter_finetune"
    "lora_finetune"
    "benign_lora_finetune"
    "embedding_attack"
)

check_model_complete() {
    local alias="$1"

    # Check all seeds
    for seed in "${SEEDS[@]}"; do
        base="results/sweeps/seed_${seed}/${alias}"

        # Check all attacks have best.json
        for attack in "${ATTACKS[@]}"; do
            best_json="${base}/${attack}/optuna_single/best.json"
            if [[ ! -f "$best_json" ]]; then
                return 1  # Not complete
            fi
        done
    done

    return 0  # Complete
}

get_model_size() {
    local cache_name="$1"
    local path="${HF_CACHE}/${cache_name}"

    if [[ -d "$path" ]]; then
        du -sh "$path" | cut -f1
    else
        echo "N/A"
    fi
}

run_cleanup_cycle() {
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Checking model completion across all seeds ==="
    echo ""

    complete_models=()
    incomplete_models=()

    for alias in "${!MODEL_CACHE_NAMES[@]}"; do
        if check_model_complete "$alias"; then
            complete_models+=("$alias")
        else
            incomplete_models+=("$alias")
        fi
    done

    echo "Complete models (${#complete_models[@]}):"
    for alias in "${complete_models[@]}"; do
        cache_name="${MODEL_CACHE_NAMES[$alias]}"
        size=$(get_model_size "$cache_name")
        echo "  $alias ($size)"
    done

    echo ""
    echo "Incomplete models (${#incomplete_models[@]}):"
    for alias in "${incomplete_models[@]}"; do
        echo "  $alias"
    done

    # Delete by default unless --dry-run flag is set
    if [[ "${#complete_models[@]}" -gt 0 ]] && [[ "$DRY_RUN" == false ]]; then
        echo ""
        echo "=== Deleting cached models for completed sweeps ==="

        total_freed=0
        for alias in "${complete_models[@]}"; do
            cache_name="${MODEL_CACHE_NAMES[$alias]}"
            path="${HF_CACHE}/${cache_name}"

            if [[ -d "$path" ]]; then
                size=$(get_model_size "$cache_name")
                echo "Deleting $alias ($size): $path"
                rm -rf "$path"
            else
                echo "Skipping $alias (already deleted or not found)"
            fi
        done

        echo "Done cleaning up."
    elif [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "[DRY RUN MODE] Would delete ${#complete_models[@]} models"
    fi

    echo ""
    echo "---"
}

# Main loop - run continuously every hour
echo "Starting continuous cleanup monitor (runs every hour)"
echo "Press Ctrl+C to stop"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN MODE: No deletions will occur"
    echo ""
fi

while true; do
    run_cleanup_cycle
    echo "Next check in 1 hour ($(date -d '+1 hour' '+%Y-%m-%d %H:%M:%S'))"
    echo ""
    sleep 3600
done
