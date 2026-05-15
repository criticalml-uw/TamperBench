#!/bin/bash
SCRIPT_DIR="/data/saad_hossain/SafeTuneBed/scripts/slurm/targeted_sweeps/seed_42"

sbatch "$SCRIPT_DIR/no_weight_modification_p1.sh"
sbatch "$SCRIPT_DIR/no_weight_modification_p2.sh"
sbatch "$SCRIPT_DIR/no_weight_modification_p3.sh"
sbatch "$SCRIPT_DIR/no_weight_modification_p4.sh"
sbatch "$SCRIPT_DIR/no_weight_modification_p5.sh"
sbatch "$SCRIPT_DIR/no_weight_modification_p6.sh"
