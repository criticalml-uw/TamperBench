#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_70b_nwm_lora

MODEL="meta-llama/Llama-3.1-70B"
ALIAS="llama3_70b_instruct"
N_TRIALS=40
RESULTS_DIR="/data/far_ai_group/saad_ws/results/sweeps/seed_42/"
CONFIGS_DIR="configs/whitebox/attacks_llama3_70b"
RANDOM_SEED=42

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/no_weight_modification" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks lora_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "Model ${ALIAS} nwm + lora complete!"
