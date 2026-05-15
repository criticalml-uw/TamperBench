#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=grid_backdoor

MODEL="meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR="results/sweep_trials/llama3_8b_instruct/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
cd ~/SafeTuneBed/

uv run scripts/whitebox/optuna_grid.py \
    "$MODEL" \
    --attacks backdoor_finetune \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
    --show-progress
