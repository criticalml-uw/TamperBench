#!/usr/bin/env python3
"""Compare best-so-far objective curves for small vs large model variants.

Reads Optuna study.db files from sweep results and plots best-so-far objective
(MAXIMIZE) for each attack, comparing a small model vs its larger counterpart.
Filters to the first N completed trials (default 30).

Produces two side-by-side subplot panels:
  Left:  Qwen3-8B vs Qwen3-32B
  Right: Llama3-8B vs Llama3-70B

Usage:
    python scripts/figures/plot_model_scale_comparison.py \\
        --sweep-dir results/sweep_2026/seed_42 \\
        --max-trials 30 \\
        -o figures/model_scale_comparison.png
"""

from __future__ import annotations

import argparse
import sqlite3
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ── Model pairs to compare ────────────────────────────────────────────────
MODEL_PAIRS: list[dict[str, Any]] = [
    {
        "title": "Qwen3: 8B vs 32B",
        "small": {"dir": "qwen3_8b", "label": "Qwen3-8B"},
        "large": {"dir": "qwen3_32b", "label": "Qwen3-32B"},
    },
    {
        "title": "Llama3: 8B vs 70B",
        "small": {"dir": "llama3_8b_instruct_baseline", "label": "Llama3-8B-Instruct"},
        "large": {"dir": "llama3_70b_instruct", "label": "Llama3-70B-Instruct"},
    },
]

ATTACK_DISPLAY: dict[str, str] = {
    "lora_finetune": "LoRA Fine-tune",
    "benign_lora_finetune": "Benign LoRA",
    "competing_objectives_finetune": "Competing Obj.",
    "full_parameter_finetune": "Full Param FT",
    "backdoor_finetune": "Backdoor FT",
    "multilingual_finetune": "Multilingual FT",
    "style_modulation_finetune": "Style Mod. FT",
    "embedding_attack": "Embedding Atk",
    "no_weight_modification": "No Modification",
}

ATTACK_COLORS: dict[str, str] = {
    "lora_finetune": "#e41a1c",
    "benign_lora_finetune": "#4daf4a",
    "competing_objectives_finetune": "#ff7f00",
    "full_parameter_finetune": "#984ea3",
    "backdoor_finetune": "#377eb8",
    "multilingual_finetune": "#a65628",
    "style_modulation_finetune": "#f781bf",
}


def load_trials(db_path: Path, max_trials: int) -> list[dict[str, Any]] | None:
    """Load first max_trials completed trials from an Optuna study.db."""
    if not db_path.exists():
        return None
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # Get completed trials ordered by trial number, limited to max_trials
    cur.execute(
        "SELECT t.trial_id, t.number FROM trials t WHERE t.number < ? ORDER BY t.number ASC",
        (max_trials,),
    )
    trials_meta = cur.fetchall()

    results = []
    for tid, num in trials_meta:
        cur.execute("SELECT value FROM trial_values WHERE trial_id=? AND objective=0", (tid,))
        row = cur.fetchone()
        obj = row[0] if row else None

        cur.execute(
            "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?",
            (tid,),
        )
        params = {}
        for pname, pval, dist_json in cur.fetchall():
            dist = json.loads(dist_json)
            if dist["name"] == "CategoricalDistribution":
                params[pname] = dist["attributes"]["choices"][int(pval)]
            else:
                params[pname] = pval
        results.append({"number": num, "objective": obj, "params": params})

    db.close()
    return results if results else None


def best_so_far(trials: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    """Compute cumulative best objective over completed trials."""
    xs, ys = [], []
    best = -float("inf")
    for t in trials:
        if t["objective"] is not None:
            best = max(best, t["objective"])
            xs.append(t["number"])
            ys.append(best)
    return xs, ys


def find_common_attacks(sweep_dir: Path, pair: dict[str, Any], max_trials: int) -> list[str]:
    """Find attacks available in both models with ≥max_trials trials."""
    common = []
    small_dir = sweep_dir / pair["small"]["dir"]
    large_dir = sweep_dir / pair["large"]["dir"]

    if not small_dir.exists() or not large_dir.exists():
        return common

    small_attacks = {p.name for p in small_dir.iterdir() if p.is_dir()}
    large_attacks = {p.name for p in large_dir.iterdir() if p.is_dir()}
    shared = small_attacks & large_attacks

    for attack in sorted(shared):
        if attack in ("no_weight_modification", "embedding_attack"):
            continue
        small_db = small_dir / attack / "optuna_single" / "study.db"
        large_db = large_dir / attack / "optuna_single" / "study.db"
        if not small_db.exists() or not large_db.exists():
            continue
        # Check both have enough trials and at least 25 completed
        skip = False
        for db_path in [small_db, large_db]:
            db = sqlite3.connect(str(db_path))
            cur = db.cursor()
            cur.execute("SELECT COUNT(*) FROM trials WHERE number < ?", (max_trials,))
            n = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM trials WHERE number < ? AND state = 'COMPLETE'",
                (max_trials,),
            )
            n_complete = cur.fetchone()[0]
            db.close()
            if n < max_trials or n_complete < 20:
                skip = True
                break
        if not skip:
            common.append(attack)
    return common


def make_summary_table(
    sweep_dir: Path,
    pair: dict[str, Any],
    attacks: list[str],
    max_trials: int,
) -> str:
    """Build a text summary table of best objectives."""
    lines = []
    header = f"{'Attack':<30s} | {pair['small']['label']:>20s} | {pair['large']['label']:>20s} | {'Delta':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for attack in attacks:
        small_trials = load_trials(
            sweep_dir / pair["small"]["dir"] / attack / "optuna_single" / "study.db",
            max_trials,
        )
        large_trials = load_trials(
            sweep_dir / pair["large"]["dir"] / attack / "optuna_single" / "study.db",
            max_trials,
        )
        small_best = max(
            (t["objective"] for t in small_trials if t["objective"] is not None),
            default=float("nan"),
        )
        large_best = max(
            (t["objective"] for t in large_trials if t["objective"] is not None),
            default=float("nan"),
        )
        delta = large_best - small_best
        disp = ATTACK_DISPLAY.get(attack, attack)
        lines.append(f"{disp:<30s} | {small_best:>20.4f} | {large_best:>20.4f} | {delta:>+8.4f}")
    return "\n".join(lines)


def plot_pair(
    ax: plt.Axes,
    sweep_dir: Path,
    pair: dict[str, Any],
    attacks: list[str],
    max_trials: int,
) -> None:
    """Plot best-so-far curves for one model pair."""
    for attack in attacks:
        color = ATTACK_COLORS.get(attack, "#999999")
        disp = ATTACK_DISPLAY.get(attack, attack)

        for variant, style, alpha in [
            ("small", "--", 0.7),
            ("large", "-", 1.0),
        ]:
            db_path = sweep_dir / pair[variant]["dir"] / attack / "optuna_single" / "study.db"
            trials = load_trials(db_path, max_trials)
            if not trials:
                continue
            xs, ys = best_so_far(trials)
            label = f"{disp} ({pair[variant]['label']})"
            ax.plot(xs, ys, linestyle=style, color=color, alpha=alpha, label=label, linewidth=1.8)

    ax.set_facecolor("white")
    ax.set_title(pair["title"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel("Best Objective (StrongREJECT SR)", fontsize=11)
    ax.set_xlim(-0.5, max_trials - 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=1)


def plot_bar_comparison(
    ax: plt.Axes,
    sweep_dir: Path,
    pair: dict[str, Any],
    attacks: list[str],
    max_trials: int,
) -> None:
    """Bar chart comparing best objective per attack for a model pair."""
    x = np.arange(len(attacks))
    width = 0.35
    small_vals, large_vals = [], []

    for attack in attacks:
        for vals, variant in [(small_vals, "small"), (large_vals, "large")]:
            db_path = sweep_dir / pair[variant]["dir"] / attack / "optuna_single" / "study.db"
            trials = load_trials(db_path, max_trials)
            best = 0.0
            if trials:
                completed = [t["objective"] for t in trials if t["objective"] is not None]
                best = max(completed) if completed else 0.0
            vals.append(best)

    bars_small = ax.bar(
        x - width / 2,
        small_vals,
        width,
        label=pair["small"]["label"],
        color="#5B9BD5",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_large = ax.bar(
        x + width / 2,
        large_vals,
        width,
        label=pair["large"]["label"],
        color="#ED7D31",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bars in [bars_small, bars_large]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    attack_labels = [ATTACK_DISPLAY.get(a, a) for a in attacks]
    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Best Objective (SR)", fontsize=11)
    ax.set_title(pair["title"], fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/sweep_2026/seed_42"),
        help="Root sweep directory containing model subdirs",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=30,
        help="Max trial number to include (first N trials)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for the figure (default: model_scale_comparison.png in sweep dir)",
    )
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    max_trials: int = args.max_trials

    # ── Find common attacks per pair ──────────────────────────────────────
    pair_attacks: list[list[str]] = []
    for pair in MODEL_PAIRS:
        attacks = find_common_attacks(sweep_dir, pair, max_trials)
        pair_attacks.append(attacks)
        print(f"\n{pair['title']}:")
        print(f"  Common attacks with ≥{max_trials} trials: {attacks}")
        print()
        print(make_summary_table(sweep_dir, pair, attacks, max_trials))

    # ── Figure 1: Combined bar chart ─────────────────────────────────────
    # Build unified x-axis: "Attack (PairTitle)" for each pair×attack combo
    all_labels: list[str] = []
    all_small: list[float] = []
    all_large: list[float] = []
    all_small_names: list[str] = []
    all_large_names: list[str] = []
    group_boundaries: list[int] = []  # index where each pair group starts

    for pair, attacks in zip(MODEL_PAIRS, pair_attacks):
        group_boundaries.append(len(all_labels))
        for attack in attacks:
            disp = ATTACK_DISPLAY.get(attack, attack)
            all_labels.append(disp)
            for vals, variant in [(all_small, "small"), (all_large, "large")]:
                db_path = sweep_dir / pair[variant]["dir"] / attack / "optuna_single" / "study.db"
                trials = load_trials(db_path, max_trials)
                best = 0.0
                if trials:
                    completed = [t["objective"] for t in trials if t["objective"] is not None]
                    best = max(completed) if completed else 0.0
                vals.append(best)
            all_small_names.append(pair["small"]["label"])
            all_large_names.append(pair["large"]["label"])

    x = np.arange(len(all_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Use different color pairs per group
    group_colors = [("#5B9BD5", "#ED7D31"), ("#7BC47F", "#C75A93")]
    for gi, (pair, attacks) in enumerate(zip(MODEL_PAIRS, pair_attacks)):
        start = group_boundaries[gi]
        end = group_boundaries[gi + 1] if gi + 1 < len(group_boundaries) else len(all_labels)
        idx = list(range(start, end))
        c_small, c_large = group_colors[gi % len(group_colors)]
        ax.bar(
            x[idx] - width / 2,
            [all_small[i] for i in idx],
            width,
            label=pair["small"]["label"],
            color=c_small,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x[idx] + width / 2,
            [all_large[i] for i in idx],
            width,
            label=pair["large"]["label"],
            color=c_large,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels
        for i in idx:
            for val, xoff in [(all_small[i], -width / 2), (all_large[i], width / 2)]:
                ax.annotate(
                    f"{val:.3f}",
                    xy=(x[i] + xoff, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Separator lines between groups
    for gi in range(1, len(group_boundaries)):
        sep_x = group_boundaries[gi] - 0.5
        ax.axvline(sep_x, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_facecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Best Objective (SR)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Model Scale Comparison — Best Objective @ {max_trials} Trials",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fig.patch.set_facecolor("none")
    out_bar = args.output or (sweep_dir / "model_scale_bar_comparison.png")
    fig.savefig(out_bar, dpi=200, bbox_inches="tight")
    print(f"\nSaved bar chart: {out_bar}")

    # ── Figure 2: Best-so-far curves ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, pair, attacks in zip(axes2, MODEL_PAIRS, pair_attacks):
        if attacks:
            plot_pair(ax, sweep_dir, pair, attacks, max_trials)
        else:
            ax.text(
                0.5,
                0.5,
                "No common attacks\nwith enough trials",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(pair["title"], fontsize=13, fontweight="bold")

    fig2.suptitle(
        f"Best-so-far Objective Curves — First {max_trials} Trials",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig2.tight_layout()

    fig2.patch.set_facecolor("none")
    out_curve = out_bar.with_name(out_bar.stem + "_curves" + out_bar.suffix)
    fig2.savefig(out_curve, dpi=200, bbox_inches="tight")
    print(f"Saved curves:    {out_curve}")


if __name__ == "__main__":
    main()
