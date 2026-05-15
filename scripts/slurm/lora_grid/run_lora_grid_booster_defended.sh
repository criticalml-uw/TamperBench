#!/bin/bash
# Run all lora_finetune grid configs on the booster-defended Qwen3-8B checkpoint

DEFENDED_MODEL="/data/far_ai_group/saad_ws/results/defense_sweeps/qwen3_8b/booster/optuna_single/trial_22/defended_model"
RESULTS_DIR="results/single_2026_03_22"
MODEL_ALIAS="qwen3_8b_booster_defended"

for config in qwen3_8b qwen3_8b_a qwen3_8b_b qwen3_8b_c; do
    echo "=== Running $config on booster-defended model ==="
    uv run python scripts/whitebox/run_single_attack.py "$DEFENDED_MODEL" \
        --attack lora_finetune \
        --config-name "$config" \
        --results-dir "$RESULTS_DIR" \
        --model-alias "$MODEL_ALIAS" \
        --cleanup-checkpoints
done
