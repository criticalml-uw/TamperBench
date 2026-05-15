#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/tar_orig"

sbatch "$SCRIPT_DIR/llama3_8b_base_8gpu.sh"
sbatch "$SCRIPT_DIR/llama3_8b_instruct_8gpu.sh"
sbatch "$SCRIPT_DIR/qwen3_8b_8gpu.sh"
