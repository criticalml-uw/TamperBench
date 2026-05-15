#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=ctrl_llama3_8b_inst

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/ctrl/harden.py --model "meta-llama/Meta-Llama-3-8B-Instruct"

echo "CTRL hardening complete for meta-llama/Meta-Llama-3-8B-Instruct"

huggingface-cli upload sdhossain24/Meta-Llama-3-8B-Instruct-CTRL data/ctrl_hardened/meta-llama_Meta-Llama-3-8B-Instruct/hardened_model
echo "Uploaded to sdhossain24/Meta-Llama-3-8B-Instruct-CTRL"
