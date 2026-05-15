#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=5
#SBATCH --gpus-per-node=5
#SBATCH --time=24:00:00
#SBATCH --partition=tamper_resistance
#SBATCH --job-name=qwen3_32b_trial


cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-32B \
    --attacks no_weight_modification benign_lora_finetune benign_full_parameter_finetune lora_finetune full_parameter_finetune competing_objectives_finetune backdoor_finetune style_modulation_finetune embedding_attack \
    --results_dir results/qwen3_32_from_8 \
    --configs-dir results/nov7_trial/aggregated_eps200/qwen3_8b/

uv run scripts/whitebox/optuna_single.py \
    Qwen/Qwen3-32B \
    --attacks lora_finetune \
    --n_trials 40 \
    --results_dir results/nov7_trial/qwen3_32b/ \
    --configs-dir configs/whitebox/attacks_qwen

find "results/nov7_trial/qwen3_32b/lora_finetune" -type d -name "safetunebed_model_checkpoint" -exec rm -rf {} +
