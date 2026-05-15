#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42"

# Tuned champs (booster/crl — commented out)
sbatch "$SCRIPT_DIR/llama3_8b_booster_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_crl_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_booster_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_crl_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_booster_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_crl_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_base_booster_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_base_crl_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_base_tar_lora_low_poison.sh"

# Non-tuned HF-hosted (tar_v2 — commented out)
sbatch "$SCRIPT_DIR/llama3_8b_tar_v2_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_tar_v2_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_tar_v2_non_tuned_lora_low_poison.sh"

# Tuned champs (tar)
sbatch "$SCRIPT_DIR/llama3_8b_tar_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_tar_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_tar_lora_low_poison.sh"

# Non-tuned HF-hosted (booster/crl)
sbatch "$SCRIPT_DIR/llama3_8b_booster_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_crl_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_booster_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_crl_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_booster_non_tuned_lora_low_poison.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_crl_non_tuned_lora_low_poison.sh"
