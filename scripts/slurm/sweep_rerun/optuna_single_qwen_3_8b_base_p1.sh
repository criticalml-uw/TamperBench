#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=qwen3_8b_base_p1

MODEL="Qwen/Qwen3-8B-Base"
N_TRIALS=20
RESULTS_DIR="results/oct29_trial/qwen3_8b_base/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks no_weight_modification \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}no_weight_modification" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks benign_full_parameter_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}benign_full_parameter_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \;

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks benign_lora_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}benign_lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks full_parameter_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}full_parameter_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \;

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks lora_finetune \
    --n_trials $N_TRIALS \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks embedding_attack \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +
