#!/bin/bash
# Submit the SDD replication pipeline as SLURM jobs on the CAIS cluster.
#
# Pipeline:
#   Phase 1: Harden model with SDD defense
#   Phase 2a: MFT attack on SDD-hardened model (10, 50, 100-shot)
#   Phase 2b: MFT attack on vanilla model (10, 50, 100-shot) [baselines]
#
# attack.py runs evals (StrongREJECT + MMLU-Pro) as part of benchmark().
#
# Usage:
#   bash scripts/sdd/submit_slurm.sh [--dry-run] [--tier TIER]
#
# Example:
#   bash scripts/sdd/submit_slurm.sh
#   bash scripts/sdd/submit_slurm.sh --dry-run
#   bash scripts/sdd/submit_slurm.sh --tier minimal

set -euo pipefail

# --- Cluster-specific paths (edit for your environment) ---
REPO_DIR="${SDD_REPO_DIR:-/data/tom_tseng/TamperBench}"
UV="${SDD_UV:-/data/tom_tseng/.local/bin/uv}"
PARTITION="${SDD_PARTITION:-tamper_resistance}"

DRY_RUN=false
TIER="llama2_chat"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --tier) TIER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LOG_DIR="${REPO_DIR}/data/sdd_hardened/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "=== SDD Replication SLURM Submission ==="
echo "Tier: ${TIER}"
echo "Partition: ${PARTITION}"
echo ""

# --- Phase 1: Harden with SDD ---
HARDEN_SCRIPT="#!/bin/bash
#SBATCH --job-name=sdd_harden_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/harden_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/harden_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/sdd/harden.py --tier ${TIER}"

# --- Eval-only: Initial (no attack) baselines ---
EVAL_VANILLA_SCRIPT="#!/bin/bash
#SBATCH --job-name=sdd_eval_vanilla_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_vanilla_initial_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_vanilla_initial_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/sdd/evaluate.py --tier ${TIER}"

EVAL_SDD_SCRIPT="#!/bin/bash
#SBATCH --job-name=sdd_eval_sdd_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/eval_sdd_initial_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/eval_sdd_initial_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/sdd/evaluate.py --tier ${TIER} --sdd"

# --- Phase 2: Attacks (SDD-defended + vanilla baselines) ---
# Each k-shot attack runs independently after harden completes.
SHOTS=(10 50 100)

declare -a ATTACK_SCRIPTS
declare -a ATTACK_LABELS

for K in "${SHOTS[@]}"; do
    # SDD-defended attack
    ATTACK_SCRIPTS+=("#!/bin/bash
#SBATCH --job-name=sdd_atk_def_${K}shot_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/attack_sdd_${K}shot_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/attack_sdd_${K}shot_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/sdd/attack.py --tier ${TIER} --num-harmful ${K}")
    ATTACK_LABELS+=("SDD ${K}-shot")

    # Vanilla baseline attack
    ATTACK_SCRIPTS+=("#!/bin/bash
#SBATCH --job-name=sdd_atk_van_${K}shot_${TIER}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/attack_vanilla_${K}shot_${TIER}_%j.out
#SBATCH --error=${LOG_DIR}/attack_vanilla_${K}shot_${TIER}_%j.err

cd ${REPO_DIR}
export WANDB_MODE=disabled
${UV} run python scripts/sdd/attack.py --tier ${TIER} --num-harmful ${K} --no-defense")
    ATTACK_LABELS+=("Vanilla ${K}-shot")
done

if $DRY_RUN; then
    echo "--- Phase 1: harden ---"
    echo "$HARDEN_SCRIPT"
    echo ""
    echo "--- Eval: Vanilla initial ---"
    echo "$EVAL_VANILLA_SCRIPT"
    echo ""
    echo "--- Eval: SDD initial (depends on harden) ---"
    echo "$EVAL_SDD_SCRIPT"
    echo ""
    for i in "${!ATTACK_SCRIPTS[@]}"; do
        echo "--- Phase 2: ${ATTACK_LABELS[$i]} ---"
        echo "${ATTACK_SCRIPTS[$i]}"
        echo ""
    done
else
    HARDEN_JOB=$(echo "$HARDEN_SCRIPT" | sbatch --parsable)
    echo "Submitted Phase 1 (harden): job ${HARDEN_JOB}"

    ALL_JOBS="${HARDEN_JOB}"

    # Eval-only baselines
    EVAL_VAN_JOB=$(echo "$EVAL_VANILLA_SCRIPT" | sbatch --parsable)
    echo "Submitted Eval (Vanilla initial): job ${EVAL_VAN_JOB}"
    ALL_JOBS="${ALL_JOBS} ${EVAL_VAN_JOB}"

    EVAL_SDD_JOB=$(echo "$EVAL_SDD_SCRIPT" | sbatch --parsable --dependency=afterok:${HARDEN_JOB})
    echo "Submitted Eval (SDD initial):     job ${EVAL_SDD_JOB}  (depends on ${HARDEN_JOB})"
    ALL_JOBS="${ALL_JOBS} ${EVAL_SDD_JOB}"

    for i in "${!ATTACK_SCRIPTS[@]}"; do
        LABEL="${ATTACK_LABELS[$i]}"
        SCRIPT="${ATTACK_SCRIPTS[$i]}"

        # Vanilla attacks don't depend on harden; SDD attacks do
        if [[ "$LABEL" == Vanilla* ]]; then
            JOB=$(echo "$SCRIPT" | sbatch --parsable)
        else
            JOB=$(echo "$SCRIPT" | sbatch --parsable --dependency=afterok:${HARDEN_JOB})
        fi
        echo "Submitted Phase 2 (${LABEL}): job ${JOB}"
        ALL_JOBS="${ALL_JOBS} ${JOB}"
    done

    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all:   scancel ${ALL_JOBS}"
fi
