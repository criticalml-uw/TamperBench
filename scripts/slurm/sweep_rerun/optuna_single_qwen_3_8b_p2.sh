#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=qwen3_8b_p2

MODEL="Qwen/Qwen3-8B"
N_TRIALS=20
RESULTS_DIR="results/oct29_trial/qwen3_8b/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks multilingual_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}multilingual_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks competing_objectives_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}competing_objectives_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \;

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks backdoor_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}backdoor_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks style_modulation_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}style_modulation_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \;
