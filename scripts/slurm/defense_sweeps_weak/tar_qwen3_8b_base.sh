#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=tar_qwen3_8b_base_weak

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/defense_sweep.py Qwen/Qwen3-8B-Base \
    --defense tar \
    --n-trials 30 \
    --model-alias qwen3_8b_base \
    --results-dir /data/far_ai_group/saad_ws/results/defense_sweeps_weak \
    --random-seed 42 \
    --keep-checkpoints \
    --sweep-config single_objective_sweep_weak_qwen3_8b_base.yaml

find /data/far_ai_group/saad_ws/results/defense_sweeps_weak/qwen3_8b_base -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "TAR weak defense sweep complete for Qwen3-8B-Base!"
