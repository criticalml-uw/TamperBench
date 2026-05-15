#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=llama3_8b_mistral_7b

MODEL="meta-llama/Meta-Llama-3-8B"
N_TRIALS=40
RESULTS_DIR="results/nov7_trial/llama3_8b_baseline/"
CONFIGS_DIR="configs/whitebox/attacks_llama"

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

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
N_TRIALS=40
RESULTS_DIR="results/nov7_trial/llama3_8b_instruct_baseline/"
CONFIGS_DIR="configs/whitebox/attacks_llama"

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

MODEL="mistralai/Mistral-7B-v0.1"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/mistral_7b_base/"
CONFIGS_DIR="configs/whitebox/attacks"

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

MODEL="mistralai/Mistral-7B-Instruct-v0.1"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/mistral_7b_instruct/"
CONFIGS_DIR="configs/whitebox/attacks"

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
