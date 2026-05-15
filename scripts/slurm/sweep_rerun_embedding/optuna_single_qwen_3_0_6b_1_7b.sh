#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=qwen3_0_6b_1_7b

MODEL="Qwen/Qwen3-0.6B-Base"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/qwen3_0_6b_base/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"


uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks embedding_attack \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +

# ------

MODEL="Qwen/Qwen3-0.6B"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/qwen3_0_6b/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"


uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks embedding_attack \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +

# ------

MODEL="Qwen/Qwen3-1.7B-Base"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/qwen3_1_7b_base/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"


uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks embedding_attack \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +

# ------

MODEL="Qwen/Qwen3-1.7B"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/qwen3_1_7b/"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"


uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks embedding_attack \
    --n_trials 1 \
    --results_dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR"

find "${RESULTS_DIR}embedding_attack" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +
