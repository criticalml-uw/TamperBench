#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_8b_triplet_adv_p2

MODEL="samuelsimko/Meta-Llama-3-8B-Instruct-Triplet-Adv"
ALIAS="llama3_8b_triplet_adv"
N_TRIALS=40
RESULTS_DIR="results/sweep_2026/seed_22/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
RANDOM_SEED=22

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks competing_objectives_finetune full_parameter_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/competing_objectives_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find "${RESULTS_DIR}${ALIAS}/full_parameter_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "Model ${ALIAS} part 2 complete!"
