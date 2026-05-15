#!/bin/bash
# Run all lora_finetune grid configs on the undefended Qwen3-8B

RESULTS_DIR="results/single_2026_03_22"
MODEL_ALIAS="qwen3_8b_undefended"

for config in qwen3_8b qwen3_8b_a qwen3_8b_b qwen3_8b_c; do
    echo "=== Running $config on undefended model ==="
    uv run python scripts/whitebox/run_single_attack.py Qwen/Qwen3-8B \
        --attack lora_finetune \
        --config-name "$config" \
        --results-dir "$RESULTS_DIR" \
        --model-alias "$MODEL_ALIAS" \
        --cleanup-checkpoints
done
