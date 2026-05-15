#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=booster_qwen3_8b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/run_defense.py \
    Qwen/Qwen3-8B \
    --defense booster \
    --config_name qwen3_8b \
    --output_dir /data/far_ai_group/saad_ws/results/defense_checkpoints/booster_qwen3_8b

echo "Booster defense complete for Qwen/Qwen3-8B"
