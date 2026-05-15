#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=nwm_p1

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"
RESULTS_DIR="/data/far_ai_group/saad_ws/results/targeted_sweeps_2/seed_42/"

echo "=== llama3_8b_baseline ==="
uv run scripts/whitebox/optuna_single.py \
    "meta-llama/Meta-Llama-3-8B" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_baseline" --random-seed 42

echo "=== llama3_8b_instruct_baseline ==="
uv run scripts/whitebox/optuna_single.py \
    "meta-llama/Meta-Llama-3-8B-Instruct" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_instruct_baseline" --random-seed 42

echo "=== qwen3_8b ==="
uv run scripts/whitebox/optuna_single.py \
    "Qwen/Qwen3-8B" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b" --random-seed 42

echo "=== llama3_8b_booster_non_tuned ==="
uv run scripts/whitebox/optuna_single.py \
    "sdhossain24/Meta-Llama-3-8B-Booster" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_booster_non_tuned" --random-seed 42

echo "no_weight_modification part 1 complete!"
