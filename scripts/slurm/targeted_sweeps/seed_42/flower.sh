#!/bin/bash
# Master submission script for low-poison LoRA finetune sweeps (seed 42, poison_ratio=0.02)
# Models: Qwen3-8B, Qwen3-8B-Base, Llama-3-8B, Llama-3-8B-Instruct

sbatch /data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42/qwen3_8b_lora_low_poison.sh
sbatch /data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42/qwen3_8b_base_lora_low_poison.sh
sbatch /data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42/llama3_8b_baseline_lora_low_poison.sh
sbatch /data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42/llama3_8b_instruct_baseline_lora_low_poison.sh
