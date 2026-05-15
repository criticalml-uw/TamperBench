#!/bin/bash
# Submit all 84 jobs (21 models x 4 parts) - seed 22

cd "$(dirname "$0")"

# Large models llama (8B)
sbatch llama3_8b_baseline_p1.sh
sbatch llama3_8b_baseline_p2.sh
sbatch llama3_8b_baseline_p3.sh
sbatch llama3_8b_baseline_p4.sh
sbatch llama3_8b_instruct_baseline_p1.sh
sbatch llama3_8b_instruct_baseline_p2.sh
sbatch llama3_8b_instruct_baseline_p3.sh
sbatch llama3_8b_instruct_baseline_p4.sh

# Defended models
# sbatch llama3_8b_rr_p1.sh
# sbatch llama3_8b_rr_p2.sh
# sbatch llama3_8b_rr_p3.sh
# sbatch llama3_8b_rr_p4.sh
# sbatch llama3_8b_refat_p1.sh
# sbatch llama3_8b_refat_p2.sh
# sbatch llama3_8b_refat_p3.sh
# sbatch llama3_8b_refat_p4.sh
# sbatch llama3_8b_triplet_adv_p1.sh
# sbatch llama3_8b_triplet_adv_p2.sh
# sbatch llama3_8b_triplet_adv_p3.sh
# sbatch llama3_8b_triplet_adv_p4.sh
# sbatch llama3_8b_lat_p1.sh
# sbatch llama3_8b_lat_p2.sh
# sbatch llama3_8b_lat_p3.sh
# sbatch llama3_8b_lat_p4.sh
# sbatch llama3_8b_tar_p1.sh
# sbatch llama3_8b_tar_p2.sh
# sbatch llama3_8b_tar_p3.sh
# sbatch llama3_8b_tar_p4.sh

# Large models non-llama (7-8B)
# sbatch mistral_7b_base_p1.sh
# sbatch mistral_7b_base_p2.sh
# sbatch mistral_7b_base_p3.sh
# sbatch mistral_7b_base_p4.sh
# sbatch mistral_7b_instruct_p1.sh
# sbatch mistral_7b_instruct_p2.sh
# sbatch mistral_7b_instruct_p3.sh
# sbatch mistral_7b_instruct_p4.sh
sbatch qwen3_8b_base_p1.sh
sbatch qwen3_8b_base_p2.sh
sbatch qwen3_8b_base_p3.sh
sbatch qwen3_8b_base_p4.sh
sbatch qwen3_8b_p1.sh
sbatch qwen3_8b_p2.sh
sbatch qwen3_8b_p3.sh
sbatch qwen3_8b_p4.sh

# Medium models (3-4B)
sbatch llama3_3b_base_p1.sh
sbatch llama3_3b_base_p2.sh
sbatch llama3_3b_base_p3.sh
sbatch llama3_3b_base_p4.sh
sbatch llama3_3b_instruct_p1.sh
sbatch llama3_3b_instruct_p2.sh
sbatch llama3_3b_instruct_p3.sh
sbatch llama3_3b_instruct_p4.sh
sbatch qwen3_4b_base_p1.sh
sbatch qwen3_4b_base_p2.sh
sbatch qwen3_4b_base_p3.sh
sbatch qwen3_4b_base_p4.sh
sbatch qwen3_4b_p1.sh
sbatch qwen3_4b_p2.sh
sbatch qwen3_4b_p3.sh
sbatch qwen3_4b_p4.sh

# Small models (<3B)
sbatch qwen3_0_6b_base_p1.sh
sbatch qwen3_0_6b_base_p2.sh
sbatch qwen3_0_6b_base_p3.sh
sbatch qwen3_0_6b_base_p4.sh
sbatch qwen3_0_6b_p1.sh
sbatch qwen3_0_6b_p2.sh
sbatch qwen3_0_6b_p3.sh
sbatch qwen3_0_6b_p4.sh
sbatch llama3_1b_base_p1.sh
sbatch llama3_1b_base_p2.sh
sbatch llama3_1b_base_p3.sh
sbatch llama3_1b_base_p4.sh
sbatch llama3_1b_instruct_p1.sh
sbatch llama3_1b_instruct_p2.sh
sbatch llama3_1b_instruct_p3.sh
sbatch llama3_1b_instruct_p4.sh
sbatch qwen3_1_7b_base_p1.sh
sbatch qwen3_1_7b_base_p2.sh
sbatch qwen3_1_7b_base_p3.sh
sbatch qwen3_1_7b_base_p4.sh
sbatch qwen3_1_7b_p1.sh
sbatch qwen3_1_7b_p2.sh
sbatch qwen3_1_7b_p3.sh
sbatch qwen3_1_7b_p4.sh

echo "Submitted 84 jobs (21 models x 4 parts) - seed 22"
