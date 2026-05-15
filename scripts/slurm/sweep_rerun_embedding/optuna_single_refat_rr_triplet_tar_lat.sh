#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=refat_rr_crl_tar_lat

MODEL="samuelsimko/Meta-Llama-3-8B-Instruct-ReFAT"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/llama3_8b_refat/"
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

MODEL="GraySwanAI/Llama-3-8B-Instruct-RR"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/llama3_8b_rr/"
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

MODEL="samuelsimko/Meta-Llama-3-8B-Instruct-Triplet-Adv"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/llama3_8b_triplet_adv/"
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

MODEL="sdhossain24/lat-llama3-8b-instruct-rt-jailbreak-robust1"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/llama3_8b_lat/"
CONFIGS_DIR="configs/whitebox/attacks_llama3_8_custom"

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

MODEL="lapisrocks/Llama-3-8B-Instruct-TAR-Refusal"
N_TRIALS=20
RESULTS_DIR="results/nov7_trial/llama3_8b_tar/"
CONFIGS_DIR="configs/whitebox/attacks_llama3_8_custom"

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
