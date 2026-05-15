#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=grid_backdoor

MODEL="Qwen/Qwen3-8B"
RESULTS_DIR="results/sweep_trials/qwen3_8b/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"
cd ~/SafeTuneBed/

uv run scripts/whitebox/optuna_grid.py \
    "$MODEL" \
    --attacks backdoor_finetune \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
    --show-progress
