#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42"

sbatch "$SCRIPT_DIR/llama3_8b_booster_lora.sh"
sbatch "$SCRIPT_DIR/llama3_8b_crl_lora.sh"
sbatch "$SCRIPT_DIR/llama3_8b_tar_lora.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_booster_lora.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_crl_lora.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_tar_lora.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_booster_lora.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_crl_lora.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_tar_lora.sh"
