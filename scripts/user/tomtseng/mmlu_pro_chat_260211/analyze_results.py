"""Analyze MMLU-Pro prompt format variant results.

Compares 4 prompt format variants across all model-attack combinations:
- few-shot + continuation (baseline)
- zero-shot + continuation
- few-shot + chat template
- zero-shot + chat template

Reports:
- Which variant is best for each model-attack pair
- Whether chat models consistently prefer chat templates
- Overall statistics on which variant wins most often

Usage:
    uv run scripts/user/tomtseng/mmlu_pro_chat/analyze_results.py \
        --results-dir results/mmlu_pro_chat
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tamperbench.utils import get_repo_root

REPO_ROOT = get_repo_root()

VARIANTS = ["fewshot_continuation", "zeroshot_continuation", "fewshot_chat", "zeroshot_chat"]

# Models that are instruct/chat-tuned (vs base)
INSTRUCT_MODELS = {
    "llama3_1b_instruct",
    "llama3_3b_instruct",
    "llama3_8b_instruct",
    "llama3_8b_instruct_baseline",
    "mistral_7b_instruct",
    "qwen3_0_6b",  # Qwen3 non-base = instruct
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_8b",
    # Defense models are instruct-based
    "llama3_8b_lat",
    "llama3_8b_refat",
    "llama3_8b_rr",
    "llama3_8b_tar",
    "llama3_8b_triplet_adv",
}

BASE_MODELS = {
    "llama3_1b_base",
    "llama3_3b_base",
    "llama3_8b_base",
    "llama3_8b_baseline",
    "mistral_7b_base",
    "qwen3_0_6b_base",
    "qwen3_1_7b_base",
    "qwen3_4b_base",
    "qwen3_8b_base",
}


def load_accuracy(eval_json_path: Path) -> float | None:
    """Load accuracy from an evaluation.json file."""
    if not eval_json_path.exists():
        return None
    with open(eval_json_path) as f:
        data = json.load(f)
    # evaluation.json format: [{"metric_name": "mmlu_pro_accuracy", "metric_value": 0.xx}]
    for row in data:
        if row.get("metric_name") == "mmlu_pro_accuracy":
            return float(row["metric_value"])
    return None


def collect_results(results_dir: Path) -> list[dict[str, Any]]:
    """Collect all results into a flat list of records."""
    records: list[dict[str, Any]] = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "logs":
            continue
        model_alias = model_dir.name

        for attack_dir in sorted(model_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            attack_name = attack_dir.name

            # Find trial directories
            for trial_dir in sorted(attack_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue

                variants_dir = trial_dir / "mmlu_pro_variants"
                if not variants_dir.exists():
                    continue

                record: dict[str, Any] = {
                    "model": model_alias,
                    "attack": attack_name,
                    "is_instruct": model_alias in INSTRUCT_MODELS,
                    "is_base": model_alias in BASE_MODELS,
                }

                for variant in VARIANTS:
                    # Try current name first, then legacy name
                    eval_json = variants_dir / variant / "tamperbench_evaluation" / "mmlu_pro_val" / "evaluation.json"
                    if not eval_json.exists():
                        eval_json = (
                            variants_dir / variant / "safetunebed_evaluation" / "mmlu_pro_val" / "evaluation.json"
                        )
                    acc = load_accuracy(eval_json)
                    record[variant] = acc

                # Only include if at least one variant has results
                if any(record.get(v) is not None for v in VARIANTS):
                    records.append(record)

    return records


def analyze(records: list[dict[str, Any]]) -> None:
    """Print analysis of results."""
    if not records:
        print("No results found.")
        return

    print(f"Total model-attack pairs with results: {len(records)}")
    print()

    # Count completeness
    complete = [r for r in records if all(r.get(v) is not None for v in VARIANTS)]
    print(f"Pairs with all 4 variants: {len(complete)}")
    for v in VARIANTS:
        count = sum(1 for r in records if r.get(v) is not None)
        print(f"  {v}: {count} results")
    print()

    if not complete:
        print("No complete results to analyze.")
        return

    # Which variant wins most often?
    win_counts: dict[str, float] = defaultdict(float)
    win_counts_instruct: dict[str, float] = defaultdict(float)
    win_counts_base: dict[str, float] = defaultdict(float)

    for r in complete:
        best_variant = max(VARIANTS, key=lambda v: r[v])
        best_score = r[best_variant]
        # Count ties
        winners = [v for v in VARIANTS if r[v] == best_score]
        for w in winners:
            win_counts[w] += 1 / len(winners)
            if r["is_instruct"]:
                win_counts_instruct[w] += 1 / len(winners)
            if r["is_base"]:
                win_counts_base[w] += 1 / len(winners)

    print("=== Best variant (win count across all model-attack pairs) ===")
    for v in VARIANTS:
        print(f"  {v}: {win_counts[v]:.1f} wins ({win_counts[v] / len(complete) * 100:.1f}%)")
    print()

    if any(r["is_instruct"] for r in complete):
        instruct_complete = [r for r in complete if r["is_instruct"]]
        print(f"=== Instruct/chat models only ({len(instruct_complete)} pairs) ===")
        for v in VARIANTS:
            print(
                f"  {v}: {win_counts_instruct[v]:.1f} wins ({win_counts_instruct[v] / len(instruct_complete) * 100:.1f}%)"
            )
        print()

    if any(r["is_base"] for r in complete):
        base_complete = [r for r in complete if r["is_base"]]
        print(f"=== Base models only ({len(base_complete)} pairs) ===")
        for v in VARIANTS:
            print(f"  {v}: {win_counts_base[v]:.1f} wins ({win_counts_base[v] / len(base_complete) * 100:.1f}%)")
        print()

    # Average accuracy per variant
    print("=== Average accuracy per variant ===")
    for v in VARIANTS:
        scores = [r[v] for r in complete]
        avg = sum(scores) / len(scores)
        print(f"  {v}: {avg:.4f}")
    print()

    # Average accuracy per variant, by model type
    predicates: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("instruct", lambda r: r["is_instruct"]),
        ("base", lambda r: r["is_base"]),
    ]
    for model_type, predicate in predicates:
        subset = [r for r in complete if predicate(r)]
        if not subset:
            continue
        print(f"=== Average accuracy ({model_type} models, n={len(subset)}) ===")
        for v in VARIANTS:
            scores = [r[v] for r in subset]
            avg = sum(scores) / len(scores)
            print(f"  {v}: {avg:.4f}")
        print()

    # Improvement of chat template over continuation
    print("=== Chat template improvement over continuation ===")
    for shots_label, shot_count_variants in [
        ("few-shot", ("fewshot_chat", "fewshot_continuation")),
        ("zero-shot", ("zeroshot_chat", "zeroshot_continuation")),
    ]:
        chat_v, cont_v = shot_count_variants
        diffs = [r[chat_v] - r[cont_v] for r in complete]
        avg_diff = sum(diffs) / len(diffs)
        positive = sum(1 for d in diffs if d > 0)
        print(f"  {shots_label}: avg diff = {avg_diff:+.4f}, chat better in {positive}/{len(diffs)} cases")
    print()

    # Per-model breakdown
    print("=== Per-model average accuracy ===")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in complete:
        by_model[r["model"]].append(r)

    # Also collect no_weight_modification (unattacked) results per model
    nwm_by_model: dict[str, dict[str, float | None]] = {}
    for r in complete:
        if r["attack"] == "no_weight_modification":
            nwm_by_model[r["model"]] = {v: r[v] for v in VARIANTS}

    header = f"{'Model':<35}" + "".join(f"{v:<25}" for v in VARIANTS) + "Best"
    print(header)
    print("-" * len(header))
    for model in sorted(by_model.keys()):
        model_records = by_model[model]
        avgs: dict[str, float] = {}
        for v in VARIANTS:
            avgs[v] = sum(r[v] for r in model_records) / len(model_records)
        best = max(VARIANTS, key=lambda v: avgs[v])
        row = f"{model:<35}"
        for v in VARIANTS:
            marker = " *" if v == best else ""
            row += f"{avgs[v]:.4f}{marker:<20}"
        row += best
        print(row)

    print()
    print("=== Per-model no_weight_modification (unattacked) accuracy ===")
    print(header)
    print("-" * len(header))
    for model in sorted(nwm_by_model.keys()):
        nwm = nwm_by_model[model]
        best = max(VARIANTS, key=lambda v: nwm[v] or 0.0)
        row = f"{model:<35}"
        for v in VARIANTS:
            val = nwm[v]
            if val is not None:
                marker = " *" if v == best else ""
                row += f"{val:.4f}{marker:<20}"
            else:
                row += f"{'N/A':<25}"
        row += best
        print(row)


def main() -> None:
    """Analyze MMLU-Pro prompt format variant results."""
    parser = argparse.ArgumentParser(description="Analyze MMLU-Pro prompt format variant results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mmlu_pro_chat",
        help="Directory containing evaluation results",
    )
    args = parser.parse_args()

    records = collect_results(args.results_dir)
    analyze(records)


if __name__ == "__main__":
    main()
