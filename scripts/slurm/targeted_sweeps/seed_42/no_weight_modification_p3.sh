#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=nwm_p3

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"
RESULTS_DIR="/data/far_ai_group/saad_ws/results/targeted_sweeps_2/seed_42/"

echo "=== llama3_8b_instruct_tar_v2_non_tuned ==="
uv run scripts/whitebox/optuna_single.py \
    "sdhossain24/Meta-Llama-3-8B-Instruct-TAR" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_instruct_tar_v2_non_tuned" --random-seed 42

echo "=== qwen3_8b_booster_non_tuned ==="
uv run scripts/whitebox/optuna_single.py \
    "sdhossain24/Qwen3-8B-Booster" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b_booster_non_tuned" --random-seed 42

echo "=== qwen3_8b_crl_non_tuned ==="
uv run scripts/whitebox/optuna_single.py \
    "sdhossain24/Qwen3-8B-CRL" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b_crl_non_tuned" --random-seed 42

echo "=== qwen3_8b_tar_v2_non_tuned ==="
uv run scripts/whitebox/optuna_single.py \
    "sdhossain24/Qwen3-8B-TAR" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b_tar_v2_non_tuned" --random-seed 42

echo "no_weight_modification part 3 complete!"
