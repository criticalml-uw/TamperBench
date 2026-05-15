#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=crl_llama3_8b_inst_weak

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/defense_sweep.py meta-llama/Meta-Llama-3-8B-Instruct \
    --defense crl \
    --n-trials 30 \
    --model-alias llama3_8b_instruct \
    --results-dir /data/far_ai_group/saad_ws/results/defense_sweeps_weak \
    --random-seed 42 \
    --sweep-config single_objective_sweep_weak_llama3_8b_instruct.yaml

find /data/far_ai_group/saad_ws/results/defense_sweeps_weak/llama3_8b_instruct -type d -name "defended_model" -exec rm -rf {} + 2>/dev/null || true
find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "CRL weak defense sweep complete for Llama-3-8B-Instruct!"
