#!/bin/bash
# Submit all 120 jobs (15 defense models x 8 parts) - seed 42
# Uses whitebox_old configs (b17fae0-era attack parameters)

cd "$(dirname "$0")"

# === Llama-3-8B base defended models ===

# Booster
for i in {1..8}; do sbatch llama3_8b_booster_p${i}.sh; done

# CRL
for i in {1..8}; do sbatch llama3_8b_crl_p${i}.sh; done

# CTRL
# for i in {1..8}; do sbatch llama3_8b_ctrl_p${i}.sh; done

# RSN-Tune
# for i in {1..8}; do sbatch llama3_8b_rsn_tune_p${i}.sh; done

# TAR
for i in {1..8}; do sbatch llama3_8b_tar_v2_p${i}.sh; done

# === Llama-3-8B-Instruct defended models ===

# Instruct Booster
for i in {1..8}; do sbatch llama3_8b_instruct_booster_p${i}.sh; done

# Instruct CRL
for i in {1..8}; do sbatch llama3_8b_instruct_crl_p${i}.sh; done

# Instruct CTRL
# for i in {1..8}; do sbatch llama3_8b_instruct_ctrl_p${i}.sh; done

# Instruct RSN-Tune
# for i in {1..8}; do sbatch llama3_8b_instruct_rsn_tune_p${i}.sh; done

# Instruct TAR
for i in {1..8}; do sbatch llama3_8b_instruct_tar_v2_p${i}.sh; done

# === Qwen3-8B defended models ===

# Qwen Booster
for i in {1..8}; do sbatch qwen3_8b_booster_p${i}.sh; done

# Qwen CRL
for i in {1..8}; do sbatch qwen3_8b_crl_p${i}.sh; done

# Qwen CTRL
# for i in {1..8}; do sbatch qwen3_8b_ctrl_p${i}.sh; done

# Qwen RSN-Tune
# for i in {1..8}; do sbatch qwen3_8b_rsn_tune_p${i}.sh; done

# Qwen TAR
for i in {1..8}; do sbatch qwen3_8b_tar_v2_p${i}.sh; done

echo "Submitted 120 jobs (15 defense models x 8 parts) - seed 42"
