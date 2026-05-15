#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=rsn_llama3_8b_inst

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="data/rsn_tune_hardened/meta-llama_Meta-Llama-3-8B-Instruct"

uv run scripts/rsn_tune/harden.py \
    --model "$MODEL" \
    --output "$OUTPUT_DIR"

echo "RSN-Tune hardening complete for $MODEL"

huggingface-cli upload sdhossain24/Meta-Llama-3-8B-Instruct-RSN-Tuned "$OUTPUT_DIR"
echo "Uploaded to sdhossain24/Meta-Llama-3-8B-Instruct-RSN-Tuned"
