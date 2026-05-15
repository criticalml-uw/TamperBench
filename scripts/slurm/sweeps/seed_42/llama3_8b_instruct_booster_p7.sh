#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_8b_instruct_booster_p7

MODEL="sdhossain24/Meta-Llama-3-8B-Instruct-Booster"
ALIAS="llama3_8b_instruct_booster"
N_TRIALS=40
RESULTS_DIR="results/sweeps/seed_42/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
RANDOM_SEED=42

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks lora_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true


uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks refusal_ablation --n-trials 10 \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/refusal_ablation" -type d -name "tamperbench_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true

echo "Model ${ALIAS} part 7 complete!"
