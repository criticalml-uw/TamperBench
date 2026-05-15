#!/usr/bin/env python3
"""Generate heatmap visualizations from strong_reject_rubric evaluation files.

This script reads rubric evaluation files from strong_reject_rubric directories
and generates heatmap visualizations for all rubric metrics.

Usage:
    uv run python scripts/strong_reject_rubric/generate_rubric_heatmaps.py \
        results/sweeps/seed_42/aggregated_eps10

    # Also include rubric_plus_plus metrics (after running generate_rubric_plus.py)
    uv run python scripts/strong_reject_rubric/generate_rubric_heatmaps.py \
        results/sweeps/seed_42/aggregated_eps10 --include-plus
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# --- Model ordering (left to right) ---

MODELS_ORDER: list[str] = [
    "qwen3_0_6b_base",
    "qwen3_0_6b",
    "llama3_1b_base",
    "llama3_1b_instruct",
    "qwen3_1_7b_base",
    "qwen3_1_7b",
    "llama3_3b_base",
    "llama3_3b_instruct",
    "qwen3_4b_base",
    "qwen3_4b",
    "mistral_7b_base",
    "mistral_7b_instruct",
    "qwen3_8b_base",
    "qwen3_8b",
    "llama3_8b_baseline",
    "llama3_8b_instruct_baseline",
    "llama3_8b_rr",
    "llama3_8b_refat",
    "llama3_8b_triplet_adv",
    "llama3_8b_lat",
    "llama3_8b_tar",
    # Defense models
    "llama3_8b_booster",
    "llama3_8b_crl",
    "llama3_8b_ctrl",
    "llama3_8b_rsn_tune",
    "llama3_8b_tar_v2",
    "llama3_8b_instruct_booster",
    "llama3_8b_instruct_crl",
    "llama3_8b_instruct_ctrl",
    "llama3_8b_instruct_rsn_tune",
    "llama3_8b_instruct_tar_v2",
    "qwen3_8b_booster",
    "qwen3_8b_crl",
    "qwen3_8b_ctrl",
    "qwen3_8b_rsn_tune",
    "qwen3_8b_tar_v2",
]

# --- Attack categories ---

BASELINE_ATTACK = "no_weight_modification"

STEALTHY_ATTACKS: list[str] = [
    "backdoor_finetune",
    "competing_objectives_finetune",
    "style_modulation_finetune",
]

DIRECT_ATTACKS: list[str] = [
    "full_parameter_finetune",
    "lora_finetune",
    "multilingual_finetune",
]

BENIGN_ATTACKS: list[str] = [
    "benign_full_parameter_finetune",
    "benign_lora_finetune",
]

EMBEDDING_ATTACK = "embedding_attack"

# --- Metric configurations ---

METRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "strong_reject_score_gemma": {
        "display_name": "SR Score (Gemma)",
        "short_name": "SR-Gemma",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric",
    },
    "strong_reject_score_rubric": {
        "display_name": "SR Score (Rubric)",
        "short_name": "SR-Rubric",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric",
    },
    "refusal": {
        "display_name": "Refusal Rate",
        "short_name": "Refusal",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Blues",
        "cmap_range": (0.02, 0.75),
        "use_min": True,
        "source_dir": "strong_reject_rubric",
    },
    "convincingness": {
        "display_name": "Convincingness (Non-Refused)",
        "short_name": "Conv.",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Purples",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric",
    },
    "specificity": {
        "display_name": "Specificity (Non-Refused)",
        "short_name": "Spec.",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Oranges",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric",
    },
    # Plus-plus metrics (only available after running generate_rubric_plus.py)
    "convincingness_non_refused": {
        "display_name": "Convincingness (Non-Refused)",
        "short_name": "Conv-NR",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Purples",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric_plus_plus",
    },
    "specificity_non_refused": {
        "display_name": "Specificity (Non-Refused)",
        "short_name": "Spec-NR",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Oranges",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric_plus_plus",
    },
    "convincingness_refused": {
        "display_name": "Convincingness (Refused)",
        "short_name": "Conv-R",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Purples",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric_plus_plus",
    },
    "specificity_refused": {
        "display_name": "Specificity (Refused)",
        "short_name": "Spec-R",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Oranges",
        "cmap_range": (0.02, 0.75),
        "source_dir": "strong_reject_rubric_plus_plus",
    },
}

# Base metrics (always generated)
BASE_METRICS: list[str] = [
    "strong_reject_score_gemma",
    "strong_reject_score_rubric",
    "refusal",
    "convincingness",
    "specificity",
]

# Plus-plus metrics (only with --include-plus)
PLUS_METRICS: list[str] = [
    "convincingness_non_refused",
    "specificity_non_refused",
    "convincingness_refused",
    "specificity_refused",
]

FloatArray = npt.NDArray[np.floating[Any]]


# --- Plotting helpers ---


def truncated_cmap(name: str, minval: float, maxval: float) -> LinearSegmentedColormap:
    """Create truncated colormap to avoid extreme light/dark values."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(minval, maxval, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_trunc", colors)
    cmap.set_bad((0.96, 0.96, 0.96, 0.0))
    return cmap


def format_value(value: float) -> str:
    """Format value for cell annotation."""
    return "" if not np.isfinite(value) else f"{value:.2f}"


def get_text_color(value: float, norm: Normalize, cmap: LinearSegmentedColormap) -> str:
    """Return white for dark cells, dark for light cells."""
    if not np.isfinite(value):
        return "#1b1b1b"
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if luminance < 0.5 else "#1b1b1b"


def display_attack_name(attack: str) -> str:
    """Format attack name for display."""
    return attack.replace("_", " ").title()


def reorder_by_models(models: list[str], matrix: FloatArray, order: list[str]) -> tuple[list[str], FloatArray]:
    """Reorder models and data columns according to preferred order."""
    model_to_idx = {m: i for i, m in enumerate(models)}
    # Include ordered models that exist in data, then any remaining
    ordered_indices = [model_to_idx[m] for m in order if m in model_to_idx]
    remaining = [i for i in range(len(models)) if i not in ordered_indices]
    indices = ordered_indices + remaining
    models_sorted = [models[i] for i in indices]
    return models_sorted, matrix[:, indices]


# --- Data collection ---


def collect_scores(
    agg_dir: Path,
    metric_name: str,
) -> tuple[dict[tuple[str, str], float], list[str], list[str]]:
    """Collect rubric scores from evaluation.json files.

    Returns:
        Tuple of (scores_dict, models_list, attacks_list)
    """
    source_dir = METRIC_CONFIGS[metric_name]["source_dir"]
    scores: dict[tuple[str, str], float] = {}
    models_set: set[str] = set()
    attacks_set: set[str] = set()

    for model_dir in sorted(agg_dir.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith(".")
            or model_dir.name.startswith("heatmap")
            or model_dir.name == "rubric_heatmaps"
        ):
            continue

        for attack_dir in sorted(model_dir.iterdir()):
            if not attack_dir.is_dir():
                continue

            # Find evaluation directory
            eval_dirs = list(attack_dir.glob("trial_*_tamperbench_evaluation"))
            if not eval_dirs:
                continue

            rubric_file = eval_dirs[0] / source_dir / "evaluation.json"
            if not rubric_file.exists():
                continue

            try:
                with open(rubric_file) as f:
                    metrics = json.load(f)

                for metric in metrics:
                    if metric["metric_name"] == metric_name:
                        value = metric["metric_value"]
                        if value is not None and np.isfinite(value):
                            scores[(model_dir.name, attack_dir.name)] = value
                            models_set.add(model_dir.name)
                            attacks_set.add(attack_dir.name)
                        break

            except Exception as e:
                LOGGER.warning(f"Error reading {model_dir.name}/{attack_dir.name}: {e}")

    return scores, sorted(models_set), sorted(attacks_set)


def build_matrix(
    scores: dict[tuple[str, str], float],
    models: list[str],
    attacks: list[str],
) -> FloatArray:
    """Build (n_attacks, n_models) matrix from scores dict."""
    matrix = np.full((len(attacks), len(models)), np.nan)
    for i, attack in enumerate(attacks):
        for j, model in enumerate(models):
            if (model, attack) in scores:
                matrix[i, j] = scores[(model, attack)]
    return matrix


def build_grouped_rows(
    matrix: FloatArray,
    attacks: list[str],
    short_name: str,
    use_min: bool = False,
) -> tuple[list[str], FloatArray]:
    """Build grouped rows: baseline, attacks by category, averages, overall max/min."""
    attack_to_idx = {a: i for i, a in enumerate(attacks)}
    rows: list[FloatArray] = []
    labels: list[str] = []
    n_models = matrix.shape[1]

    # Baseline
    if BASELINE_ATTACK in attack_to_idx:
        rows.append(matrix[attack_to_idx[BASELINE_ATTACK]])
        labels.append(f"{display_attack_name(BASELINE_ATTACK)} {short_name}")

    # Stealthy + Direct
    group_rows: list[FloatArray] = []
    for attack in STEALTHY_ATTACKS + DIRECT_ATTACKS:
        if attack in attack_to_idx:
            rows.append(matrix[attack_to_idx[attack]])
            labels.append(f"{display_attack_name(attack)} {short_name}")
            group_rows.append(matrix[attack_to_idx[attack]])

    if group_rows:
        avg: FloatArray = np.nanmean(np.stack(group_rows), axis=0)
        rows.append(avg)
        labels.append(f"Avg: Stealthy + Direct {short_name}")

    # Benign
    benign_rows: list[FloatArray] = []
    for attack in BENIGN_ATTACKS:
        if attack in attack_to_idx:
            rows.append(matrix[attack_to_idx[attack]])
            labels.append(f"{display_attack_name(attack)} {short_name}")
            benign_rows.append(matrix[attack_to_idx[attack]])

    if benign_rows:
        avg = np.nanmean(np.stack(benign_rows), axis=0)
        rows.append(avg)
        labels.append(f"Avg: Benign {short_name}")

    # Embedding attack
    if EMBEDDING_ATTACK in attack_to_idx:
        rows.append(matrix[attack_to_idx[EMBEDDING_ATTACK]])
        labels.append(f"{display_attack_name(EMBEDDING_ATTACK)} {short_name}")

    # Overall max or min (across all non-baseline attacks)
    all_attack_rows: list[FloatArray] = []
    for attack in attacks:
        if attack != BASELINE_ATTACK and attack in attack_to_idx:
            all_attack_rows.append(matrix[attack_to_idx[attack]])
    if all_attack_rows:
        stacked = np.stack(all_attack_rows)
        if use_min:
            overall: FloatArray = np.nanmin(stacked, axis=0)
            labels.append(f"Overall Min {short_name}")
        else:
            overall = np.nanmax(stacked, axis=0)
            labels.append(f"Overall Max {short_name}")
        rows.append(overall)

    if not rows:
        return labels, np.full((0, n_models), np.nan)

    return labels, np.stack(rows)


# --- Plotting ---


def plot_metric_heatmap(
    matrix: FloatArray,
    labels: list[str],
    models: list[str],
    metric_name: str,
    output_path: Path,
) -> None:
    """Plot heatmap for a single rubric metric."""
    config = METRIC_CONFIGS[metric_name]
    n_rows = len(labels)
    n_models = len(models)

    cmap = truncated_cmap(config["cmap"], config["cmap_range"][0], config["cmap_range"][1])
    norm = Normalize(vmin=config["vmin"], vmax=config["vmax"])

    fig_width = max(18.0, 0.9 * n_models)
    fig_height = max(5.4, 0.54 * n_rows)

    fig: Figure
    axes: npt.NDArray[Any]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.2, right=0.82, top=0.93, bottom=0.08, hspace=0.0)
    fig.suptitle(config["display_name"], fontsize=14, fontweight="bold", y=0.97)

    for row_idx, ax in enumerate(axes):
        ax: Axes
        row_data = matrix[row_idx, :][np.newaxis, :]
        ax.imshow(row_data, cmap=cmap, norm=norm, aspect="auto")

        ax.set_yticks([])
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_xticks(range(n_models))

        if row_idx == n_rows - 1:
            ax.set_xticklabels(models, rotation=90, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        label = labels[row_idx]
        is_summary = label.startswith("Avg:") or label.startswith("Overall")
        ax.set_ylabel(
            label,
            rotation=0,
            labelpad=18,
            fontsize=8,
            ha="right",
            va="center",
            fontweight="bold" if is_summary else "normal",
        )

        for col_idx in range(n_models):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    0,
                    format_value(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold" if is_summary else "normal",
                    color=get_text_color(value, norm, cmap),
                )

    # Colorbar
    cbar_ax = fig.add_axes((0.84, 0.15, 0.03, 0.7))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(config["display_name"], rotation=270, labelpad=14, fontsize=9)
    ticks = list(np.linspace(config["vmin"], config["vmax"], 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate rubric metric heatmaps from evaluation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "agg_dir",
        type=Path,
        help="Aggregated epsilon directory (e.g., results/sweeps/seed_42/aggregated_eps10)",
    )
    parser.add_argument(
        "--include-plus",
        action="store_true",
        help="Include rubric_plus_plus metrics (convincingness/specificity by refusal status)",
    )

    args = parser.parse_args()

    if not args.agg_dir.exists():
        LOGGER.error(f"Directory does not exist: {args.agg_dir}")
        return

    output_dir = args.agg_dir / "rubric_heatmaps"
    output_dir.mkdir(exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    metrics_to_process = list(BASE_METRICS)
    if args.include_plus:
        metrics_to_process += PLUS_METRICS

    for metric_name in metrics_to_process:
        config = METRIC_CONFIGS[metric_name]
        LOGGER.info(f"Processing: {config['display_name']}")

        scores, models, attacks = collect_scores(args.agg_dir, metric_name)
        LOGGER.info(f"  Found {len(scores)} model/attack combinations")

        if not scores:
            LOGGER.warning(f"  No scores found, skipping")
            continue

        # Build (n_attacks, n_models) matrix
        raw_matrix = build_matrix(scores, models, attacks)

        # Reorder models
        models, raw_matrix = reorder_by_models(models, raw_matrix, MODELS_ORDER)

        # Build grouped rows with averages and overall max/min
        use_min = config.get("use_min", False)
        labels, grouped_matrix = build_grouped_rows(raw_matrix, attacks, config["short_name"], use_min=use_min)

        # Plot
        output_path = output_dir / f"heatmap_rubric_{metric_name}.png"
        plot_metric_heatmap(grouped_matrix, labels, models, metric_name, output_path)

    LOGGER.info(f"Done! All heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
