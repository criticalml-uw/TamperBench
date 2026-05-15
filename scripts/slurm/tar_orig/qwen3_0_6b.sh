#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=tar_orig_qwen3_0_6b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run python scripts/tar/test_tar.py "Qwen/Qwen3-0.6B" \
    --results-dir /data/far_ai_group/saad_ws/results/tar_orig/qwen3_0_6b \
    --num-gpus 2

echo "TAR orig complete for qwen3_0_6b!"
