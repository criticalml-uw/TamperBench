#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=tar_orig_llama3_1b_instruct

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run python scripts/tar/test_tar.py "meta-llama/Llama-3.2-1B-Instruct" \
    --results-dir /data/far_ai_group/saad_ws/results/tar_orig/llama3_1b_instruct \
    --num-gpus 2

echo "TAR orig complete for llama3_1b_instruct!"
