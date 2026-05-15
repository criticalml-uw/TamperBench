#!/bin/bash
# Submit all 168 jobs (21 models x 8 parts) - seed 42

cd "$(dirname "$0")"

# Large models llama (8B)
# for i in {1..8}; do sbatch llama3_8b_baseline_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_baseline_p${i}.sh; done

# Defended models
# for i in {1..8}; do sbatch llama3_8b_rr_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_refat_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_triplet_adv_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_lat_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_tar_p${i}.sh; done

# RSN-Tune
# for i in {1..8}; do sbatch llama3_8b_rsn_tune_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_rsn_tune_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_rsn_tune_p${i}.sh; done

# SDD defense
# for i in {1..8}; do sbatch llama3_8b_sdd_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_sdd_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_sdd_p${i}.sh; done

# T-Vaccine defense
# for i in {1..8}; do sbatch llama3_8b_t_vaccine_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_t_vaccine_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_t_vaccine_p${i}.sh; done


# Booster
# for i in {1..8}; do sbatch llama3_8b_booster_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_booster_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_booster_p${i}.sh; done

# CTRL
# for i in {1..8}; do sbatch qwen3_8b_ctrl_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_ctrl_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_ctrl_p${i}.sh; done

# CRL
# for i in {1..8}; do sbatch llama3_8b_crl_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_crl_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_crl_p${i}.sh; done

# TAR v2
# for i in {1..8}; do sbatch llama3_8b_tar_v2_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_8b_instruct_tar_v2_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_tar_v2_p${i}.sh; done

# Large models non-llama (7-8B)
# for i in {1..8}; do sbatch mistral_7b_base_p${i}.sh; done
# for i in {1..8}; do sbatch mistral_7b_instruct_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_base_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_8b_p${i}.sh; done

# Medium models (3-4B)
# for i in {1..8}; do sbatch llama3_3b_base_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_3b_instruct_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_4b_base_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_4b_p${i}.sh; done

# Small models (<3B)
# for i in {1..8}; do sbatch qwen3_0_6b_base_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_0_6b_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_1b_base_p${i}.sh; done
# for i in {1..8}; do sbatch llama3_1b_instruct_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_1_7b_base_p${i}.sh; done
# for i in {1..8}; do sbatch qwen3_1_7b_p${i}.sh; done

# TAR-O defense
for i in {1..8}; do sbatch llama3_8b_tar_o_p${i}.sh; done
for i in {1..8}; do sbatch llama3_8b_instruct_tar_o_p${i}.sh; done
for i in {1..8}; do sbatch qwen3_8b_tar_o_p${i}.sh; done
