#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=grid_atk_booster_non_tuned_qwen3_8b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

MODEL="sdhossain24/Qwen3-8B-Booster"
RESULTS_DIR="/data/far_ai_group/saad_ws/results/defense_champs_grid_attack"
MODEL_ALIAS="qwen3_8b_booster_non_tuned"

echo "Running attack 1/3 on qwen3_8b/booster_non_tuned (qwen3_8b_weak_copy_1)..."
uv run python scripts/whitebox/run_single_attack.py "$MODEL" \
    --attack lora_finetune \
    --config-name qwen3_8b_weak_copy_1 \
    --results-dir "$RESULTS_DIR" \
    --model-alias "$MODEL_ALIAS" \
    --random-seed 42 \
    --cleanup-checkpoints

echo "Running attack 2/3 on qwen3_8b/booster_non_tuned (qwen3_8b_weak_copy_2)..."
uv run python scripts/whitebox/run_single_attack.py "$MODEL" \
    --attack lora_finetune \
    --config-name qwen3_8b_weak_copy_2 \
    --results-dir "$RESULTS_DIR" \
    --model-alias "$MODEL_ALIAS" \
    --random-seed 42 \
    --cleanup-checkpoints

echo "Running attack 3/3 on qwen3_8b/booster_non_tuned (qwen3_8b_weak_copy_3)..."
uv run python scripts/whitebox/run_single_attack.py "$MODEL" \
    --attack lora_finetune \
    --config-name qwen3_8b_weak_copy_3 \
    --results-dir "$RESULTS_DIR" \
    --model-alias "$MODEL_ALIAS" \
    --random-seed 42 \
    --cleanup-checkpoints

find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 \
    -exec rm -rf {} \; 2>/dev/null || true

echo "Grid attack complete for qwen3_8b/booster_non_tuned!"
