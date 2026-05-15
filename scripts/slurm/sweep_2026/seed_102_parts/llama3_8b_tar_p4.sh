#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_8b_tar_p4

MODEL="lapisrocks/Llama-3-8B-Instruct-TAR-Refusal"
ALIAS="llama3_8b_tar"
N_TRIALS=40
RESULTS_DIR="results/sweep_2026/seed_102/"
CONFIGS_DIR="configs/whitebox/attacks_llama3_8_custom"
RANDOM_SEED=102

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks lora_finetune benign_lora_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find "${RESULTS_DIR}${ALIAS}/benign_lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks embedding_attack --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true

echo "Model ${ALIAS} part 4 complete!"
