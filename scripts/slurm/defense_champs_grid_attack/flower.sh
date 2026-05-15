#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/defense_champs_grid_attack"

# Non-tuned HF-hosted (tar_v2 — commented out)
# sbatch "$SCRIPT_DIR/tar_v2_non_tuned_llama3_8b.sh"
# sbatch "$SCRIPT_DIR/tar_v2_non_tuned_llama3_8b_instruct.sh"
# sbatch "$SCRIPT_DIR/tar_v2_non_tuned_qwen3_8b.sh"

# Tuned champs (tar)
# sbatch "$SCRIPT_DIR/tar_llama3_8b.sh"
# sbatch "$SCRIPT_DIR/tar_llama3_8b_instruct.sh"
# sbatch "$SCRIPT_DIR/tar_qwen3_8b.sh"
# sbatch "$SCRIPT_DIR/tar_qwen3_8b_base.sh"

# Baseline (undefended) models
# sbatch "$SCRIPT_DIR/baseline_llama3_8b.sh"
# sbatch "$SCRIPT_DIR/baseline_llama3_8b_instruct.sh"
# sbatch "$SCRIPT_DIR/baseline_qwen3_8b.sh"
# sbatch "$SCRIPT_DIR/baseline_qwen3_8b_base.sh"

# Non-tuned HF-hosted (booster/crl)
sbatch "$SCRIPT_DIR/booster_non_tuned_llama3_8b.sh"
sbatch "$SCRIPT_DIR/crl_non_tuned_llama3_8b.sh"
sbatch "$SCRIPT_DIR/booster_non_tuned_llama3_8b_instruct.sh"
sbatch "$SCRIPT_DIR/crl_non_tuned_llama3_8b_instruct.sh"
sbatch "$SCRIPT_DIR/booster_non_tuned_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/crl_non_tuned_qwen3_8b.sh"

# Tuned champs (booster/crl — commented out)
sbatch "$SCRIPT_DIR/booster_llama3_8b.sh"
sbatch "$SCRIPT_DIR/crl_llama3_8b.sh"
sbatch "$SCRIPT_DIR/booster_llama3_8b_instruct.sh"
sbatch "$SCRIPT_DIR/crl_llama3_8b_instruct.sh"
sbatch "$SCRIPT_DIR/booster_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/crl_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/booster_qwen3_8b_base.sh"
sbatch "$SCRIPT_DIR/crl_qwen3_8b_base.sh"