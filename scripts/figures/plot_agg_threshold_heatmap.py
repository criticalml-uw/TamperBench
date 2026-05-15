#!/usr/bin/env python3
"""Plot aggregated threshold heatmap — stacks SR averages at different epsilon thresholds.

Rows:
  - Untampered (1 row, from any epsilon — same across all)
  - Avg: Malicious SR at < 10% MMLU drop (eps10)
  - Avg: Malicious SR at < 20% MMLU drop (eps20)
  - Avg: Malicious SR no threshold (eps200)
  - Avg: Benign SR at < 10% MMLU drop (eps10)
  - Avg: Benign SR at < 20% MMLU drop (eps20)
  - Avg: Benign SR no threshold (eps200)
  - Embedding Attack (1 row, same across all)

Usage:
    uv run python scripts/figures/plot_agg_threshold_heatmap.py \
        results/sweeps/seed_42

    uv run python scripts/figures/plot_agg_threshold_heatmap.py \
        results/sweeps/seed_42 -o figures/heatmap_thresholds.png
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
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

MALICIOUS_ATTACKS: list[str] = [
    "backdoor_finetune",
    "competing_objectives_finetune",
    "style_modulation_finetune",
    "full_parameter_finetune",
    "lora_finetune",
    "multilingual_finetune",
]

BENIGN_ATTACKS: list[str] = [
    "benign_full_parameter_finetune",
    "benign_lora_finetune",
]

EMBEDDING_ATTACK = "embedding_attack"

# --- Epsilon directories ---

EPSILON_DIRS: list[tuple[str, str]] = [
    ("aggregated_eps10", "< 10% MMLU Drop"),
    ("aggregated_eps20", "< 20% MMLU Drop"),
    ("aggregated_eps200", "No Threshold"),
]

# --- Colormap settings ---

CMAP_NAME = "magma_r"
CMAP_MIN = 0.02
CMAP_MAX = 0.75

FloatArray = npt.NDArray[np.floating[Any]]


# --- Helpers ---


def truncated_cmap(name: str, minval: float, maxval: float) -> LinearSegmentedColormap:
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


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def compute_avg(
    sr_raw: FloatArray,
    attacks: list[str],
    attack_subset: list[str],
    model_indices: list[int],
) -> FloatArray:
    """Compute average SR across attack_subset for given model indices."""
    row = np.full(len(model_indices), np.nan)
    subset_rows: list[FloatArray] = []
    for attack in attack_subset:
        if attack in attacks:
            idx = attacks.index(attack)
            subset_rows.append(sr_raw[np.array(model_indices), idx])
    if subset_rows:
        row = np.nanmean(np.stack(subset_rows), axis=0)
    return row


def get_attack_row(
    sr_raw: FloatArray,
    attacks: list[str],
    attack_name: str,
    model_indices: list[int],
) -> FloatArray:
    """Get single attack row for given model indices."""
    row = np.full(len(model_indices), np.nan)
    if attack_name in attacks:
        idx = attacks.index(attack_name)
        row = sr_raw[np.array(model_indices), idx]
    return row


# --- Main ---


def plot_threshold_heatmap(sweep_dir: Path, output_path: Path) -> None:
    """Build and plot the aggregated threshold heatmap."""
    # Load all epsilon JSONs
    eps_data: dict[str, dict[str, Any]] = {}
    for eps_dir_name, _label in EPSILON_DIRS:
        json_path = sweep_dir / eps_dir_name / "heatmap_max_sr.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue
        eps_data[eps_dir_name] = load_json(json_path)

    if not eps_data:
        print("No data found")
        return

    # Use model ordering — filter to models in MODELS_ORDER that exist in at least one JSON
    all_model_sets = [set(d["models"]) for d in eps_data.values()]
    all_models_union = set().union(*all_model_sets)
    models = [m for m in MODELS_ORDER if m in all_models_union]
    n_models = len(models)

    # Build rows
    rows: list[FloatArray] = []
    labels: list[str] = []
    right_labels: list[str] = []

    # Untampered — use eps200 (unbounded) as source, all should be identical
    ref_key = list(eps_data.keys())[-1]  # last = eps200
    ref = eps_data[ref_key]
    ref_model_to_idx = {m: i for i, m in enumerate(ref["models"])}
    ref_model_indices = [ref_model_to_idx[m] for m in models if m in ref_model_to_idx]
    ref_models_available = [m for m in models if m in ref_model_to_idx]

    # Build a full-width untampered row (NaN for missing models)
    untampered = np.full(n_models, np.nan)
    for i, m in enumerate(models):
        if m in ref_model_to_idx:
            untampered[i] = get_attack_row(
                np.array(ref["safety_raw"]), ref["attacks"], BASELINE_ATTACK, [ref_model_to_idx[m]]
            )[0]
    rows.append(untampered)
    labels.append("Untampered")
    right_labels.append("")

    # Malicious avg at each threshold
    for eps_dir_name, threshold_label in EPSILON_DIRS:
        if eps_dir_name not in eps_data:
            continue
        d = eps_data[eps_dir_name]
        d_model_to_idx = {m: i for i, m in enumerate(d["models"])}

        row = np.full(n_models, np.nan)
        for i, m in enumerate(models):
            if m in d_model_to_idx:
                val = compute_avg(np.array(d["safety_raw"]), d["attacks"], MALICIOUS_ATTACKS, [d_model_to_idx[m]])
                row[i] = val[0]
        rows.append(row)
        labels.append("Avg: Malicious")
        right_labels.append(threshold_label)

    # Benign avg at each threshold
    for eps_dir_name, threshold_label in EPSILON_DIRS:
        if eps_dir_name not in eps_data:
            continue
        d = eps_data[eps_dir_name]
        d_model_to_idx = {m: i for i, m in enumerate(d["models"])}

        row = np.full(n_models, np.nan)
        for i, m in enumerate(models):
            if m in d_model_to_idx:
                val = compute_avg(np.array(d["safety_raw"]), d["attacks"], BENIGN_ATTACKS, [d_model_to_idx[m]])
                row[i] = val[0]
        rows.append(row)
        labels.append("Avg: Benign")
        right_labels.append(threshold_label)

    # Embedding attack — same across thresholds, use eps200
    embedding = np.full(n_models, np.nan)
    for i, m in enumerate(models):
        if m in ref_model_to_idx:
            embedding[i] = get_attack_row(
                np.array(ref["safety_raw"]), ref["attacks"], EMBEDDING_ATTACK, [ref_model_to_idx[m]]
            )[0]
    rows.append(embedding)
    labels.append("Embedding Attack")
    right_labels.append("")

    matrix = np.stack(rows)
    n_rows = len(labels)

    # --- Plot ---
    cmap = truncated_cmap(CMAP_NAME, CMAP_MIN, CMAP_MAX)
    norm = Normalize(vmin=0, vmax=1.0)

    fig_width = max(18.0, 0.9 * n_models)
    fig_height = max(5.4, 0.6 * n_rows)

    fig: Figure
    axes: npt.NDArray[Any]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.14, right=0.82, top=0.95, bottom=0.10, hspace=0.0)

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

        # Left label
        label = labels[row_idx]
        is_bold = label.startswith("Avg:")
        ax.set_ylabel(
            label,
            rotation=0,
            labelpad=14,
            fontsize=9,
            ha="right",
            va="center",
            fontweight="bold" if is_bold else "normal",
        )

        # Right label (threshold)
        if right_labels[row_idx]:
            ax.text(
                n_models - 0.3,
                0,
                right_labels[row_idx],
                ha="left",
                va="center",
                fontsize=8,
                fontstyle="italic",
                color="#444444",
                transform=ax.transData,
            )

        # Cell annotations
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
                    fontweight="bold" if is_bold else "normal",
                    color=get_text_color(value, norm, cmap),
                )

    # Thick horizontal lines between groups: after untampered, after malicious, after benign
    group_boundaries = [0, 3, 6]  # indices of last row in each group
    for boundary_idx in group_boundaries:
        if boundary_idx < n_rows:
            axes[boundary_idx].spines["bottom"].set_visible(True)
            axes[boundary_idx].spines["bottom"].set_color("#202020")
            axes[boundary_idx].spines["bottom"].set_linewidth(2.0)

    # Colorbar
    cbar_ax = fig.add_axes((0.84, 0.15, 0.03, 0.7))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("StrongReject", rotation=270, labelpad=14)
    ticks = list(np.linspace(0.0, 1.0, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated threshold heatmap (SR at different epsilon bounds)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Sweep directory containing aggregated_eps{10,20,200} subdirs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <sweep_dir>/heatmap_agg_thresholds.png)",
    )
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: {args.sweep_dir} does not exist")
        return

    output: Path = args.output or args.sweep_dir / "heatmap_agg_thresholds.png"
    plot_threshold_heatmap(args.sweep_dir, output)


if __name__ == "__main__":
    main()
