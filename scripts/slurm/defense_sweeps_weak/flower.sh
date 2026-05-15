#!/bin/bash
# Master submission script for weak defense sweeps (30 trials each, weak attack configs)
# Defenses: CRL, Booster, TAR
# Phase 1: Llama models, Phase 2: Qwen models
# 1 job per model/defense at a time (no concurrent writes to same Optuna DB)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Phase 1: Llama models ===

# CRL (2 GPUs) - Llama
# sbatch "$SCRIPT_DIR/crl_llama3_8b.sh"
# sbatch "$SCRIPT_DIR/crl_llama3_8b_instruct.sh"

# CRL (2 GPUs) - Qwen
sbatch "$SCRIPT_DIR/crl_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/crl_qwen3_8b_base.sh"

# Booster (1 GPU) - Llama
sbatch "$SCRIPT_DIR/booster_llama3_8b.sh"
sbatch "$SCRIPT_DIR/booster_llama3_8b_instruct.sh"

# TAR (1 GPU) - Llama
sbatch "$SCRIPT_DIR/tar_llama3_8b.sh"
sbatch "$SCRIPT_DIR/tar_llama3_8b_instruct.sh"

# === Phase 2: Qwen models ===

# Booster (1 GPU) - Qwen
sbatch "$SCRIPT_DIR/booster_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/booster_qwen3_8b_base.sh"

# TAR (1 GPU) - Qwen
sbatch "$SCRIPT_DIR/tar_qwen3_8b.sh"
sbatch "$SCRIPT_DIR/tar_qwen3_8b_base.sh"
