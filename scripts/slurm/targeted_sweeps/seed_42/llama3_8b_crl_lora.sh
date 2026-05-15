#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_8b_crl_lora

MODEL="/data/far_ai_group/saad_ws/results/defense_sweeps_weak_champs/llama3_8b/crl/best_checkpoint"
ALIAS="llama3_8b_crl"
N_TRIALS=40
RESULTS_DIR="/data/far_ai_group/saad_ws/results/sweeps/seed_42/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
RANDOM_SEED=42

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" --attacks lora_finetune --n-trials $N_TRIALS \
    --results-dir "$RESULTS_DIR" --configs-dir "$CONFIGS_DIR" --model-alias "$ALIAS" --random-seed "$RANDOM_SEED"

find "${RESULTS_DIR}${ALIAS}/lora_finetune" -type d -name "tamperbench_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "Model ${ALIAS} lora_finetune complete!"
