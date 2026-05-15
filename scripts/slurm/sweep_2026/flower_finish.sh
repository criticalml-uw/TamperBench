#!/bin/bash
# Finish incomplete llama3_8b_tar runs

cd "$(dirname "$0")"

# === SEED 22 ===
echo "Submitting seed 22 llama3_8b_tar jobs..."
sbatch seed_22_parts/llama3_8b_tar_p1.sh
sbatch seed_22_parts/llama3_8b_tar_p2.sh
sbatch seed_22_parts/llama3_8b_tar_p3.sh
sbatch seed_22_parts/llama3_8b_tar_p4.sh

# === SEED 42 ===
echo "Submitting seed 42 llama3_8b_tar jobs..."
sbatch seed_42_parts/llama3_8b_tar_p1.sh
sbatch seed_42_parts/llama3_8b_tar_p2.sh
sbatch seed_42_parts/llama3_8b_tar_p3.sh
sbatch seed_42_parts/llama3_8b_tar_p4.sh

# === SEED 102 ===
echo "Submitting seed 102 llama3_8b_tar jobs..."
sbatch seed_102_parts/llama3_8b_tar_p1.sh
sbatch seed_102_parts/llama3_8b_tar_p2.sh
sbatch seed_102_parts/llama3_8b_tar_p3.sh
sbatch seed_102_parts/llama3_8b_tar_p4.sh

echo "Submitted 12 llama3_8b_tar jobs (4 parts x 3 seeds)"