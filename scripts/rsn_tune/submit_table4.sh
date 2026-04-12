#!/bin/bash
# Submit the RSN-Tune Table 4 replication experiment as a SLURM job on the
# CAIS cluster.
#
# Usage:
#   bash scripts/rsn_tune/submit_table4.sh [--dry-run] [--model MODEL]
#
# Example:
#   bash scripts/rsn_tune/submit_table4.sh
#   bash scripts/rsn_tune/submit_table4.sh --model mistralai/Mistral-7B-Instruct-v0.2
#   bash scripts/rsn_tune/submit_table4.sh --dry-run

set -euo pipefail

REPO_DIR="${TVAC_REPO_DIR:-$HOME/TamperBench-2}"
UV="${TVAC_UV:-/data/tom_tseng/.local/bin/uv}"
PARTITION="${TVAC_PARTITION:-tamper_resistance}"

DRY_RUN=false
MODEL="meta-llama/Llama-2-7b-chat-hf"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MODEL_SHORT=$(basename "${MODEL}")
RESULTS_DIR="${REPO_DIR}/results/rsn_tune_table4"
LOG_DIR="${REPO_DIR}/results/rsn_tune_table4/slurm_logs"

echo "=== RSN-Tune Table 4 SLURM Submission ==="
echo "Model: ${MODEL}"
echo "Partition: ${PARTITION}"
echo "Results: ${RESULTS_DIR}"
echo ""

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=rsn_table4_${MODEL_SHORT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/${MODEL_SHORT}_%j.out
#SBATCH --error=${LOG_DIR}/${MODEL_SHORT}_%j.err

cd ${REPO_DIR}
mkdir -p ${LOG_DIR}
export WANDB_MODE=disabled

${UV} run python scripts/rsn_tune/table4.py ${MODEL} \\
    --results-dir ${RESULTS_DIR}"

if $DRY_RUN; then
    echo "$JOB_SCRIPT"
else
    ssh cais "mkdir -p ${LOG_DIR}"
    JOB_ID=$(ssh cais "echo '$JOB_SCRIPT' | sbatch --parsable")
    echo "Submitted job: ${JOB_ID}"
    echo ""
    echo "Monitor with: ssh cais squeue -u \$USER"
    echo "Logs:         ssh cais tail -f ${LOG_DIR}/${MODEL_SHORT}_${JOB_ID}.out"
    echo "Cancel:       ssh cais scancel ${JOB_ID}"
fi
