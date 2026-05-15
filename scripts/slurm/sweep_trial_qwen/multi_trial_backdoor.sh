#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=multi_backdoor

MODEL="Qwen/Qwen3-8B"
N_TRIALS=100
RESULTS_DIR="results/sweep_trials/qwen3_8b/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"
cd ~/SafeTuneBed/

uv run scripts/whitebox/optuna_multi.py \
    "$MODEL" \
    --attacks backdoor_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
