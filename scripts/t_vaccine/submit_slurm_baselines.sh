#!/bin/bash
# Submit baseline evaluation jobs for T-Vaccine comparison.
#
# Assumes submit_slurm.sh has already run (so the alignment checkpoint exists).
# Submits three independent evaluation scenarios:
#   1. defended-only:        base model + alignment LoRA (no attack)
#   2. base:                 base model only (no LoRAs)
#   3. undefended-attacked:  attack on base model (no defense), then evaluate
#
# The defended-attacked scenario is handled by submit_slurm.sh.
#
# Usage:
#   bash scripts/t_vaccine/submit_slurm_baselines.sh [--dry-run] [--tier TIER]

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

echo "=== T-Vaccine Baseline Evaluations ==="
echo "Tier: ${TIER}"
echo "Partition: ${PARTITION}"
echo ""

# --- 1. Evaluate defended-only (no attack) ---
EVAL_DEFENDED_ONLY="#!/bin/bash
#SBATCH --job-name=tvac_eval_defended_only_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_defended_only_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_defended_only_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/evaluate.py --tier ${TIER} --scenario defended-only"

# --- 2. Evaluate base model (no defense, no attack) ---
EVAL_BASE="#!/bin/bash
#SBATCH --job-name=tvac_eval_base_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_base_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_base_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/evaluate.py --tier ${TIER} --scenario base"

# --- 3. Attack without defense, then evaluate ---
ATTACK_NO_DEFENSE="#!/bin/bash
#SBATCH --job-name=tvac_attack_nodef_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/attack_no_defense_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/attack_no_defense_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/attack.py --tier ${TIER} --no-defense"

EVAL_UNDEFENDED_ATTACKED="#!/bin/bash
#SBATCH --job-name=tvac_eval_undef_atk_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_undefended_attacked_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_undefended_attacked_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/t_vaccine/evaluate.py --tier ${TIER} --scenario undefended-attacked"

if $DRY_RUN; then
    echo "--- defended-only eval ---"
    echo "$EVAL_DEFENDED_ONLY"
    echo ""
    echo "--- base eval ---"
    echo "$EVAL_BASE"
    echo ""
    echo "--- attack without defense ---"
    echo "$ATTACK_NO_DEFENSE"
    echo ""
    echo "--- undefended-attacked eval (depends on attack) ---"
    echo "$EVAL_UNDEFENDED_ATTACKED"
else
    DEFENDED_ONLY_JOB=$(echo "$EVAL_DEFENDED_ONLY" | sbatch --parsable)
    echo "Submitted defended-only eval:      job ${DEFENDED_ONLY_JOB}"

    BASE_JOB=$(echo "$EVAL_BASE" | sbatch --parsable)
    echo "Submitted base eval:               job ${BASE_JOB}"

    ATTACK_NODEF_JOB=$(echo "$ATTACK_NO_DEFENSE" | sbatch --parsable)
    echo "Submitted attack (no defense):     job ${ATTACK_NODEF_JOB}"

    UNDEF_ATK_EVAL_JOB=$(echo "$EVAL_UNDEFENDED_ATTACKED" | sbatch --parsable --dependency=afterok:${ATTACK_NODEF_JOB})
    echo "Submitted undefended-attacked eval: job ${UNDEF_ATK_EVAL_JOB}  (depends on ${ATTACK_NODEF_JOB})"

    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all:   scancel ${DEFENDED_ONLY_JOB} ${BASE_JOB} ${ATTACK_NODEF_JOB} ${UNDEF_ATK_EVAL_JOB}"
fi
