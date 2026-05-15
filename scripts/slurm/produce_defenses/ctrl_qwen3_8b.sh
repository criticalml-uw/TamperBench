#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=ctrl_qwen3_8b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/ctrl/harden.py --model "Qwen/Qwen3-8B"

echo "CTRL hardening complete for Qwen/Qwen3-8B"

huggingface-cli upload sdhossain24/Qwen3-8B-CTRL data/ctrl_hardened/Qwen_Qwen3-8B/hardened_model
echo "Uploaded to sdhossain24/Qwen3-8B-CTRL"
