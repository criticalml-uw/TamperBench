#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=mistral_7b_instruct_p4

MODEL="mistralai/Mistral-7B-Instruct-v0.1"
ALIAS="mistral_7b_instruct"
N_TRIALS=40
RESULTS_DIR="results/sweeps/seed_102/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
RANDOM_SEED=102

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks full_parameter_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/full_parameter_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "Model ${ALIAS} part 4 complete!"
