#!/bin/bash
# Run all lora_finetune grid configs on the undefended Llama-3-8B

RESULTS_DIR="results/single_2026_03_22"
MODEL_ALIAS="llama3_8b_undefended"

for config in llama3_8b llama3_8b_a; do
    echo "=== Running $config on undefended model ==="
    uv run python scripts/whitebox/run_single_attack.py meta-llama/Meta-Llama-3-8B \
        --attack lora_finetune \
        --config-name "$config" \
        --results-dir "$RESULTS_DIR" \
        --model-alias "$MODEL_ALIAS" \
        --cleanup-checkpoints
done
