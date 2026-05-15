#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=tar_orig_llama3_8b_instruct_8gpu

cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

# Ensure the results dir is empty so test_tar.py doesn't skip steps based on stale outputs.
RESULTS_DIR="/data/far_ai_group/saad_ws/results/tar_orig/llama3_8b_instruct"
if [ -d "$RESULTS_DIR" ]; then
    echo "Clearing existing results dir: $RESULTS_DIR"
    rm -rf "$RESULTS_DIR"
fi

uv run python scripts/tar/test_tar.py "meta-llama/Meta-Llama-3-8B-Instruct" \
    --results-dir /data/far_ai_group/saad_ws/results/tar_orig/llama3_8b_instruct \
    --num-gpus 8

echo "TAR orig (8 GPU) complete for llama3_8b_instruct!"
