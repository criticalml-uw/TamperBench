#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=single_backdoor

MODEL="meta-llama/Llama-3.1-8B-Instruct"
N_TRIALS=100
RESULTS_DIR="results/rebuttal_specific/llama3_8b_instruct/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
cd ~/SafeTuneBed/

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks lora_finetune_optim \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
