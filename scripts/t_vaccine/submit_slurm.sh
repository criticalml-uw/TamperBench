#!/bin/bash
# Submit the T-Vaccine replication pipeline (harden -> attack -> evaluate)
# as three dependent SLURM jobs on the CAIS cluster.
#
# Usage:
#   bash scripts/t_vaccine/submit_slurm.sh [--dry-run] [--tier TIER]
#
# Example:
#   bash scripts/t_vaccine/submit_slurm.sh
#   bash scripts/t_vaccine/submit_slurm.sh --dry-run
#   bash scripts/t_vaccine/submit_slurm.sh --tier minimal

set -euo pipefail

# --- Cluster-specific paths (edit these for your environment) ---
REPO_DIR="${TVAC_REPO_DIR:-/data/tom_tseng/TamperBench}"
UV="${TVAC_UV:-/data/tom_tseng/.local/bin/uv}"
PARTITION="${TVAC_PARTITION:-tamper_resistance}"

DRY_RUN=false
TIER="llama2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --tier) TIER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LOG_DIR="${REPO_DIR}/data/t_vaccine_hardened/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "=== T-Vaccine SLURM Submission ==="
echo "Tier: ${TIER}"
echo "Partition: ${PARTITION}"
echo ""

# --- Phase 1: Alignment (harden) ---
HARDEN_SCRIPT="#!/bin/bash
#SBATCH --job-name=tvac_harden_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/harden_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/harden_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/harden.py --tier ${TIER}"

# --- Phase 2: Attack (fine-tuning) ---
ATTACK_SCRIPT="#!/bin/bash
#SBATCH --job-name=tvac_attack_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/attack_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/attack_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/attack.py --tier ${TIER}"

# --- Phase 3: Evaluate ---
EVAL_SCRIPT="#!/bin/bash
#SBATCH --job-name=tvac_eval_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/evaluate.py --tier ${TIER} --scenario defended-attacked"

if $DRY_RUN; then
    echo "--- Phase 1: harden ---"
    echo "$HARDEN_SCRIPT"
    echo ""
    echo "--- Phase 2: attack (depends on Phase 1) ---"
    echo "$ATTACK_SCRIPT"
    echo ""
    echo "--- Phase 3: evaluate (depends on Phase 2) ---"
    echo "$EVAL_SCRIPT"
else
    HARDEN_JOB=$(echo "$HARDEN_SCRIPT" | sbatch --parsable)
    echo "Submitted Phase 1 (harden):   job ${HARDEN_JOB}"

    ATTACK_JOB=$(echo "$ATTACK_SCRIPT" | sbatch --parsable --dependency=afterok:${HARDEN_JOB})
    echo "Submitted Phase 2 (attack):   job ${ATTACK_JOB}  (depends on ${HARDEN_JOB})"

    EVAL_JOB=$(echo "$EVAL_SCRIPT" | sbatch --parsable --dependency=afterok:${ATTACK_JOB})
    echo "Submitted Phase 3 (evaluate): job ${EVAL_JOB}  (depends on ${ATTACK_JOB})"

    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all:   scancel ${HARDEN_JOB} ${ATTACK_JOB} ${EVAL_JOB}"
fi
