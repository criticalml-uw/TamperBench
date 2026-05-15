#!/bin/bash
# Check for missing/incomplete attacks across all seeds

cd /data/saad_hossain/SafeTuneBed

SEEDS=(22 42 102)

echo "=== Checking for missing/incomplete attacks ==="
echo ""

incomplete_embedding=()

for seed in "${SEEDS[@]}"; do
    base="results/sweep_2026/seed_${seed}"
    if [[ ! -d "$base" ]]; then
        echo "SEED $seed: Directory doesn't exist"
        continue
    fi

    echo "=== SEED $seed ==="

    found=0
    missing=0

    for model_dir in "$base"/*/; do
        [[ -d "$model_dir" ]] || continue
        model=$(basename "$model_dir")

        for attack_dir in "$model_dir"/*/; do
            [[ -d "$attack_dir" ]] || continue
            attack=$(basename "$attack_dir")

            if [[ -f "${attack_dir}optuna_single/best.json" ]]; then
                found=$((found + 1))
            else
                missing=$((missing + 1))
                echo "  MISSING: ${model}/${attack}"

                # Track incomplete embedding_attack dirs
                if [[ "$attack" == "embedding_attack" ]]; then
                    incomplete_embedding+=("${attack_dir}")
                fi
            fi
        done
    done

    echo "  Complete: $found, Missing best.json: $missing"
    echo ""
done

# Print command to remove incomplete embedding_attack dirs
if [[ ${#incomplete_embedding[@]} -gt 0 ]]; then
    echo "=== Incomplete embedding_attack directories ==="
    for dir in "${incomplete_embedding[@]}"; do
        echo "  $dir"
    done
    echo ""
    echo "To remove these incomplete embedding_attack dirs, run:"
    echo ""
    echo "  $0 --clean-embedding"
    echo ""
fi

# Handle --clean-embedding flag
if [[ "$1" == "--clean-embedding" ]]; then
    echo "=== Removing incomplete embedding_attack directories ==="
    for seed in "${SEEDS[@]}"; do
        base="results/sweep_2026/seed_${seed}"
        [[ -d "$base" ]] || continue

        for model_dir in "$base"/*/; do
            [[ -d "$model_dir" ]] || continue
            emb_dir="${model_dir}embedding_attack"

            if [[ -d "$emb_dir" ]] && [[ ! -f "${emb_dir}/optuna_single/best.json" ]]; then
                echo "Removing: $emb_dir"
                rm -rf "$emb_dir"
            fi
        done
    done
    echo "Done."
fi
