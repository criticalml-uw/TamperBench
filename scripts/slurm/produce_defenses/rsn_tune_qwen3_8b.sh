#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=rsn_qwen3_8b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

MODEL="Qwen/Qwen3-8B"
OUTPUT_DIR="data/rsn_tune_hardened/Qwen_Qwen3-8B"

uv run scripts/rsn_tune/harden.py \
    --model "$MODEL" \
    --output "$OUTPUT_DIR"

echo "RSN-Tune hardening complete for $MODEL"

huggingface-cli upload sdhossain24/Qwen3-8B-RSN-Tuned "$OUTPUT_DIR"
echo "Uploaded to sdhossain24/Qwen3-8B-RSN-Tuned"
