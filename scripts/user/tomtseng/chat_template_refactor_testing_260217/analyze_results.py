"""Analyze chat-attack regression test results.

Compares evaluation metrics from the new chat-attack branch against the
baseline metrics stored in nov7_trial best.json for each model-attack pair.

Reports:
- Side-by-side comparison of old vs new metrics (strong_reject, mmlu_pro_val)
- Absolute and relative differences
- Regressions where relative change exceeds a configurable threshold

Usage:
    uv run scripts/user/tomtseng/chat_template_refactor_testing_260217/analyze_results.py \
        --source-results-dir /data/saad_hossain/SafeTuneBed/results/nov7_trial \
        --new-results-dir results/chat_template_refactor_testing_260217
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tamperbench.utils import get_repo_root
from tamperbench.whitebox.utils.names import EvalPath

REPO_ROOT = get_repo_root()

MODELS = ["qwen3_0_6b", "qwen3_0_6b_base"]

ATTACKS = [
    "backdoor_finetune",
    "benign_full_parameter_finetune",
    "benign_lora_finetune",
    "competing_objectives_finetune",
    "full_parameter_finetune",
    "lora_finetune",
    "multilingual_finetune",
    "no_weight_modification",
    "style_modulation_finetune",
]

# Eval name -> metric_name in evaluation.json
EVAL_METRIC_NAMES: dict[str, str] = {
    "strong_reject": "strong_reject_score",
    "mmlu_pro_val": "mmlu_pro_accuracy",
}


def load_old_metrics(source_results_dir: Path, model_alias: str, attack: str) -> dict[str, float] | None:
    """Load baseline metrics from best.json's top trial.

    Returns dict mapping eval_name -> metric_value, or None if not found.
    """
    best_json_path = source_results_dir / model_alias / attack / "optuna_single" / "best.json"
    if not best_json_path.exists():
        return None
    with open(best_json_path) as f:
        best = json.load(f)
    top_trial = best["top_trials"][0]
    eval_names = top_trial.get("eval_names", list(EVAL_METRIC_NAMES.keys()))
    values = top_trial["values"]
    return dict(zip(eval_names, values, strict=False))


def load_new_metrics(new_results_dir: Path, model_alias: str, attack: str) -> dict[str, float] | None:
    """Load new metrics from regression test evaluation.json files.

    Returns dict mapping eval_name -> metric_value, or None if not found.
    """
    attack_dir = new_results_dir / model_alias / attack
    if not attack_dir.exists():
        return None

    # Find trial directory (there should be exactly one)
    trial_dirs = [d for d in attack_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
    if not trial_dirs:
        return None
    trial_dir = trial_dirs[0]

    metrics: dict[str, float] = {}
    for eval_name, metric_name in EVAL_METRIC_NAMES.items():
        eval_json_path = trial_dir / EvalPath.CHECKPOINT_EVAL_DIR / eval_name / EvalPath.EVALUATION
        if not eval_json_path.exists():
            continue
        with open(eval_json_path) as f:
            data = json.load(f)
        for row in data:
            if row.get("metric_name") == metric_name:
                metrics[eval_name] = float(row["metric_value"])
                break

    return metrics if metrics else None


def analyze(
    source_results_dir: Path,
    new_results_dir: Path,
    regression_threshold: float,
) -> None:
    """Compare old vs new metrics and report regressions."""
    print(f"Source results: {source_results_dir}")
    print(f"New results:    {new_results_dir}")
    print(f"Regression threshold: {regression_threshold:.0%} relative change")
    print()

    eval_names = list(EVAL_METRIC_NAMES.keys())
    records: list[dict[str, Any]] = []
    missing_old = 0
    missing_new = 0

    for model in MODELS:
        for attack in ATTACKS:
            old_metrics = load_old_metrics(source_results_dir, model, attack)
            new_metrics = load_new_metrics(new_results_dir, model, attack)

            if old_metrics is None:
                missing_old += 1
                continue
            if new_metrics is None:
                missing_new += 1
                continue

            record: dict[str, Any] = {"model": model, "attack": attack}
            for en in eval_names:
                old_val = old_metrics.get(en)
                new_val = new_metrics.get(en)
                record[f"{en}_old"] = old_val
                record[f"{en}_new"] = new_val
                if old_val is not None and new_val is not None:
                    record[f"{en}_diff"] = new_val - old_val
                    if abs(old_val) > 1e-9:
                        record[f"{en}_rel"] = (new_val - old_val) / abs(old_val)
                    else:
                        record[f"{en}_rel"] = float("inf") if abs(new_val - old_val) > 1e-9 else 0.0
            records.append(record)

    print(f"Pairs analyzed: {len(records)}")
    print(f"Skipped (no old baseline): {missing_old}")
    print(f"Skipped (no new results):  {missing_new}")
    print()

    if not records:
        print("No results to compare.")
        return

    # Print detailed comparison table
    header = f"{'Model':<20} {'Attack':<35}"
    for _en in eval_names:
        header += f"  {'Old':>7}  {'New':>7}  {'Diff':>8}  {'Rel%':>7}"
    print(header)
    print("-" * len(header))

    regressions: list[dict[str, Any]] = []

    for r in records:
        row = f"{r['model']:<20} {r['attack']:<35}"
        is_regression = False
        for en in eval_names:
            old_val = r.get(f"{en}_old")
            new_val = r.get(f"{en}_new")
            diff = r.get(f"{en}_diff")
            rel = r.get(f"{en}_rel")

            old_str = f"{old_val:.4f}" if old_val is not None else "N/A"
            new_str = f"{new_val:.4f}" if new_val is not None else "N/A"
            diff_str = f"{diff:+.4f}" if diff is not None else "N/A"
            rel_str = f"{rel:+.1%}" if rel is not None and rel != float("inf") else "N/A"

            row += f"  {old_str:>7}  {new_str:>7}  {diff_str:>8}  {rel_str:>7}"

            if rel is not None and abs(rel) > regression_threshold:
                is_regression = True

        if is_regression:
            row += "  *** REGRESSION ***"
            regressions.append(r)
        print(row)

    print()

    # Summary statistics
    for en in eval_names:
        diffs = [r[f"{en}_diff"] for r in records if r.get(f"{en}_diff") is not None]
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs, key=abs)
            print(f"{en}: avg diff = {avg_diff:+.4f}, max abs diff = {max_diff:+.4f}")

    print()

    # Report regressions
    if regressions:
        print(f"*** {len(regressions)} REGRESSIONS DETECTED (>{regression_threshold:.0%} relative change) ***")
        for r in regressions:
            print(f"  {r['model']}/{r['attack']}")
            for en in eval_names:
                rel = r.get(f"{en}_rel")
                if rel is not None and abs(rel) > regression_threshold:
                    print(f"    {en}: {r[f'{en}_old']:.4f} -> {r[f'{en}_new']:.4f} ({rel:+.1%})")
    else:
        print("*** NO REGRESSIONS DETECTED ***")
        print("All metrics within acceptable tolerance.")


def main() -> None:
    """Analyze chat-attack regression test results."""
    parser = argparse.ArgumentParser(description="Analyze chat-attack regression test results")
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=Path("/data/saad_hossain/SafeTuneBed/results/nov7_trial"),
        help="Directory with baseline results (best.json files)",
    )
    parser.add_argument(
        "--new-results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "chat_template_refactor_testing_260217",
        help="Directory with new regression test results",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Relative change threshold to flag as regression (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    analyze(
        source_results_dir=args.source_results_dir,
        new_results_dir=args.new_results_dir,
        regression_threshold=args.regression_threshold,
    )


if __name__ == "__main__":
    main()
