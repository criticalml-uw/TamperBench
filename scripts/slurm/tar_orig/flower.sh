#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/tar_orig"

sbatch "$SCRIPT_DIR/llama3_1b_instruct.sh"
sbatch "$SCRIPT_DIR/llama3_1b_base.sh"
sbatch "$SCRIPT_DIR/qwen3_0_6b.sh"
