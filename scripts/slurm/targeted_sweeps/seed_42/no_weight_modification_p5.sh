#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=nwm_p5

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"
RESULTS_DIR="/data/far_ai_group/saad_ws/results/targeted_sweeps_2/seed_42/"

echo "=== llama3_8b_instruct_crl ==="
uv run scripts/whitebox/optuna_single.py \
    "/data/far_ai_group/saad_ws/results/defense_sweeps_weak_champs/llama3_8b_instruct/crl/best_checkpoint" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_instruct_crl" --random-seed 42

echo "=== llama3_8b_instruct_tar ==="
uv run scripts/whitebox/optuna_single.py \
    "/data/far_ai_group/saad_ws/results/defense_sweeps_weak_champs/llama3_8b_instruct/tar/best_checkpoint" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "llama3_8b_instruct_tar" --random-seed 42

echo "=== qwen3_8b_booster ==="
uv run scripts/whitebox/optuna_single.py \
    "/data/far_ai_group/saad_ws/results/defense_sweeps_weak_champs/qwen3_8b/booster/best_checkpoint" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b_booster" --random-seed 42

echo "=== qwen3_8b_crl ==="
uv run scripts/whitebox/optuna_single.py \
    "/data/far_ai_group/saad_ws/results/defense_sweeps_weak_champs/qwen3_8b/crl/best_checkpoint" --attacks no_weight_modification --n-trials 1 \
    --results-dir "$RESULTS_DIR" --configs-dir "configs/whitebox/attacks" --model-alias "qwen3_8b_crl" --random-seed 42

echo "no_weight_modification part 5 complete!"
