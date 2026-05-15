#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=tar_qwen3_8b

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/defense_sweep.py Qwen/Qwen3-8B \
    --defense tar \
    --n-trials 30 \
    --model-alias qwen3_8b \
    --results-dir /data/far_ai_group/saad_ws/results/defense_sweeps \
    --random-seed 42 \
    --keep-checkpoints

find results/defense_sweeps/tar/qwen3_8b -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "TAR defense sweep complete for Qwen3-8B!"
