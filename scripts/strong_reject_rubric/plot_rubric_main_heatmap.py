#!/usr/bin/env python3
"""Plot main rubric heatmap — mirrors scripts/figures/plot_main_heatmap.py.

Models as columns (ordered left-to-right by size), attacks as rows grouped by
category with averages and overall max/min. One heatmap per rubric metric.

Output: rubric_heatmaps/heatmap_rubric_{metric}.png

Usage:
    uv run python scripts/strong_reject_rubric/plot_rubric_main_heatmap.py \
        results/sweeps/seed_42/aggregated_eps10

    # Single metric only
    uv run python scripts/strong_reject_rubric/plot_rubric_main_heatmap.py \
        results/sweeps/seed_42/aggregated_eps10 \
        --metric strong_reject_score_rubric
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

# --- Model ordering (left to right, by size) ---

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
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.75),
        "use_min": False,
        "source_dir": "strong_reject_rubric",
    },
    "strong_reject_score_rubric": {
        "display_name": "SR Score (Rubric)",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.75),
        "use_min": False,
        "source_dir": "strong_reject_rubric",
    },
    "refusal": {
        "display_name": "Refusal Rate",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Blues",
        "cmap_range": (0.02, 0.75),
        "use_min": True,
        "source_dir": "strong_reject_rubric",
    },
    "convincingness": {
        "display_name": "Convincingness (Non-Refused)",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Purples",
        "cmap_range": (0.02, 0.75),
        "use_min": False,
        "source_dir": "strong_reject_rubric",
    },
    "specificity": {
        "display_name": "Specificity (Non-Refused)",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Oranges",
        "cmap_range": (0.02, 0.75),
        "use_min": False,
        "source_dir": "strong_reject_rubric",
    },
}

ALL_METRICS: list[str] = list(METRIC_CONFIGS.keys())

FloatArray = npt.NDArray[np.floating[Any]]


# --- Helpers ---


def truncated_cmap(name: str, minval: float, maxval: float) -> LinearSegmentedColormap:
    """Create truncated colormap to avoid extreme light/dark values."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(minval, maxval, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_trunc", colors)
    cmap.set_bad((0.96, 0.96, 0.96, 0.0))
    return cmap


def format_value(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.2f}"


def get_text_color(value: float, norm: Normalize, cmap: LinearSegmentedColormap) -> str:
    if not np.isfinite(value):
        return "#1b1b1b"
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if luminance < 0.5 else "#1b1b1b"


def display_name(attack: str) -> str:
    return attack.replace("_", " ").title()


def reorder_by_models(models: list[str], data: FloatArray, order: list[str]) -> tuple[list[str], FloatArray]:
    """Reorder and filter models (columns) to only those in order list."""
    model_to_idx = {m: i for i, m in enumerate(models)}
    indices = [model_to_idx[m] for m in order if m in model_to_idx]
    return [models[i] for i in indices], data[indices, :]


# --- Data collection ---


def collect_metric_matrix(
    agg_dir: Path,
    metric_name: str,
) -> tuple[list[str], list[str], FloatArray]:
    """Collect scores into (n_models, n_attacks) matrix from evaluation.json files.

    Returns:
        (models, attacks, matrix) where matrix is (n_models, n_attacks)
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

            eval_dirs = list(attack_dir.glob("trial_*_tamperbench_evaluation"))
            if not eval_dirs:
                continue

            rubric_file = eval_dirs[0] / source_dir / "evaluation.json"
            if not rubric_file.exists():
                continue

            try:
                with open(rubric_file) as f:
                    metrics = json.load(f)
                for m in metrics:
                    if m["metric_name"] == metric_name:
                        val = m["metric_value"]
                        if val is not None and np.isfinite(val):
                            scores[(model_dir.name, attack_dir.name)] = val
                            models_set.add(model_dir.name)
                            attacks_set.add(attack_dir.name)
                        break
            except Exception as e:
                LOGGER.warning(f"Error reading {model_dir.name}/{attack_dir.name}: {e}")

    models = sorted(models_set)
    attacks = sorted(attacks_set)

    matrix = np.full((len(models), len(attacks)), np.nan)
    model_to_idx = {m: i for i, m in enumerate(models)}
    attack_to_idx = {a: i for i, a in enumerate(attacks)}
    for (model, attack), val in scores.items():
        matrix[model_to_idx[model], attack_to_idx[attack]] = val

    return models, attacks, matrix


# --- Grouped rows ---


def build_grouped_data(
    attacks: list[str],
    sr_raw: FloatArray,
    use_min: bool = False,
) -> tuple[list[str], FloatArray]:
    """Build grouped data with baseline, categories, averages, and overall max/min.

    sr_raw is (n_models, n_attacks). Returns (labels, matrix) where matrix is (n_rows, n_models).
    """
    rows: list[FloatArray] = []
    labels: list[str] = []

    attack_to_idx = {a: i for i, a in enumerate(attacks)}

    # Baseline
    if BASELINE_ATTACK in attack_to_idx:
        idx = attack_to_idx[BASELINE_ATTACK]
        rows.append(sr_raw[:, idx])
        labels.append(display_name(BASELINE_ATTACK))

    # Stealthy + Direct
    group_rows: list[FloatArray] = []
    for attack in STEALTHY_ATTACKS + DIRECT_ATTACKS:
        if attack in attack_to_idx:
            idx = attack_to_idx[attack]
            rows.append(sr_raw[:, idx])
            labels.append(display_name(attack))
            group_rows.append(sr_raw[:, idx])

    if group_rows:
        avg: FloatArray = np.nanmean(np.stack(group_rows), axis=0)
        rows.append(avg)
        labels.append("Avg: Stealthy + Direct")

    # Benign
    benign_rows: list[FloatArray] = []
    for attack in BENIGN_ATTACKS:
        if attack in attack_to_idx:
            idx = attack_to_idx[attack]
            rows.append(sr_raw[:, idx])
            labels.append(display_name(attack))
            benign_rows.append(sr_raw[:, idx])

    if benign_rows:
        avg = np.nanmean(np.stack(benign_rows), axis=0)
        rows.append(avg)
        labels.append("Avg: Benign")

    # Embedding attack
    if EMBEDDING_ATTACK in attack_to_idx:
        idx = attack_to_idx[EMBEDDING_ATTACK]
        rows.append(sr_raw[:, idx])
        labels.append(display_name(EMBEDDING_ATTACK))

    # Overall max or min
    all_attack_rows: list[FloatArray] = []
    for attack in attacks:
        if attack != BASELINE_ATTACK and attack in attack_to_idx:
            all_attack_rows.append(sr_raw[:, attack_to_idx[attack]])
    if all_attack_rows:
        stacked = np.stack(all_attack_rows)
        if use_min:
            overall: FloatArray = np.nanmin(stacked, axis=0)
            labels.append("Overall Min")
        else:
            overall = np.nanmax(stacked, axis=0)
            labels.append("Overall Max")
        rows.append(overall)

    return labels, np.stack(rows)


# --- Plotting ---


def plot_heatmap(
    labels: list[str],
    matrix: FloatArray,
    models: list[str],
    metric_name: str,
    output_path: Path,
) -> None:
    """Plot the main-style heatmap for one rubric metric."""
    config = METRIC_CONFIGS[metric_name]
    n_rows = len(labels)
    n_models = len(models)

    cmap = truncated_cmap(config["cmap"], config["cmap_range"][0], config["cmap_range"][1])
    vmax = config["vmax"]
    norm = Normalize(vmin=config["vmin"], vmax=vmax)

    fig_width = max(18.0, 0.9 * n_models)
    fig_height = max(5.4, 0.54 * n_rows)

    fig: Figure
    axes: npt.NDArray[Any]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.2, right=0.82, top=0.95, bottom=0.08, hspace=0.0)

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
                    color=get_text_color(value, norm, cmap),
                )

    # Colorbar
    cbar_ax = fig.add_axes((0.84, 0.15, 0.03, 0.7))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(config["display_name"], rotation=270, labelpad=14)
    ticks = list(np.linspace(config["vmin"], vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path}")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot main rubric heatmaps (all models, attacks as rows)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "agg_dir",
        type=Path,
        help="Aggregated epsilon directory (e.g., results/sweeps/seed_42/aggregated_eps10)",
    )
    parser.add_argument(
        "--metric",
        choices=ALL_METRICS,
        default=None,
        help="Generate heatmap for a single metric only",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <agg_dir>/rubric_heatmaps)",
    )

    args = parser.parse_args()

    if not args.agg_dir.exists():
        LOGGER.error(f"Directory does not exist: {args.agg_dir}")
        return

    output_dir: Path = args.output_dir or args.agg_dir / "rubric_heatmaps"
    output_dir.mkdir(exist_ok=True)

    metrics = [args.metric] if args.metric else ALL_METRICS

    for metric_name in metrics:
        config = METRIC_CONFIGS[metric_name]
        LOGGER.info(f"Processing: {config['display_name']}")

        models, attacks, raw_matrix = collect_metric_matrix(args.agg_dir, metric_name)
        LOGGER.info(f"  {len(models)} models, {len(attacks)} attacks")

        if len(models) == 0:
            LOGGER.warning(f"  No data found, skipping")
            continue

        # Reorder models
        models, raw_matrix = reorder_by_models(models, raw_matrix, MODELS_ORDER)

        # Build grouped rows
        use_min = config.get("use_min", False)
        labels, grouped = build_grouped_data(attacks, raw_matrix, use_min=use_min)

        output_path = output_dir / f"heatmap_rubric_main_{metric_name}.png"
        plot_heatmap(labels, grouped, models, metric_name, output_path)

    LOGGER.info(f"Done! Heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
