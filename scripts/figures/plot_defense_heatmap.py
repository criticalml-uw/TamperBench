#!/usr/bin/env python3
"""Plot defense comparison heatmap with 3 panels (one per base model).

Each panel shows StrongReject scores across attacks for the undefended baseline
and 5 safety defenses, with MMLU (untampered) as the first column and
summary columns (Malicious Avg, Worst Case, Benign Avg, Benign Worst).

Output: heatmap_defenses.png (or custom path via -o)

Usage:
    python scripts/figures/plot_defense_heatmap.py results/sweeps/seed_42/aggregated_eps10/heatmap_max_sr.json
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# --- Configuration ---

MODEL_FAMILIES: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "Llama-3-8B",
            {
                "baseline": "llama3_8b_baseline",
                "defenses": OrderedDict(
                    [
                        ("Booster", "llama3_8b_booster"),
                        ("CRL", "llama3_8b_crl"),
                        ("CTRL", "llama3_8b_ctrl"),
                        ("RSN-Tune", "llama3_8b_rsn_tune"),
                        ("TAR v2", "llama3_8b_tar_v2"),
                    ]
                ),
            },
        ),
        (
            "Llama-3-8B-Instruct",
            {
                "baseline": "llama3_8b_instruct_baseline",
                "defenses": OrderedDict(
                    [
                        ("Booster", "llama3_8b_instruct_booster"),
                        ("CRL", "llama3_8b_instruct_crl"),
                        ("CTRL", "llama3_8b_instruct_ctrl"),
                        ("RSN-Tune", "llama3_8b_instruct_rsn_tune"),
                        ("TAR v2", "llama3_8b_instruct_tar_v2"),
                    ]
                ),
            },
        ),
        (
            "Qwen3-8B",
            {
                "baseline": "qwen3_8b",
                "defenses": OrderedDict(
                    [
                        ("Booster", "qwen3_8b_booster"),
                        ("CRL", "qwen3_8b_crl"),
                        ("CTRL", "qwen3_8b_ctrl"),
                        ("RSN-Tune", "qwen3_8b_rsn_tune"),
                        ("TAR v2", "qwen3_8b_tar_v2"),
                    ]
                ),
            },
        ),
    ]
)

# Attack categories
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

# Column order
COLUMN_ORDER: list[str] = (
    ["mmlu"]
    + [BASELINE_ATTACK]
    + MALICIOUS_ATTACKS
    + ["malicious_avg", "malicious_worst"]
    + BENIGN_ATTACKS
    + ["benign_avg", "benign_worst"]
    + [EMBEDDING_ATTACK]
)

COLUMN_LABELS: dict[str, str] = {
    "mmlu": "MMLU\n(untampered)",
    "no_weight_modification": "No Weight\nModification",
    "backdoor_finetune": "Backdoor",
    "competing_objectives_finetune": "Competing\nObjectives",
    "style_modulation_finetune": "Style\nModulation",
    "full_parameter_finetune": "Full Param\nFinetune",
    "lora_finetune": "LoRA\nFinetune",
    "multilingual_finetune": "Multilingual\nFinetune",
    "malicious_avg": "Malicious\nAvg",
    "malicious_worst": "Malicious\nWorst",
    "benign_full_parameter_finetune": "Benign Full\nParam",
    "benign_lora_finetune": "Benign\nLoRA",
    "benign_avg": "Benign\nAvg",
    "benign_worst": "Benign\nWorst",
    "embedding_attack": "Embedding\nAttack",
}

# Which columns are summary/aggregate columns
SUMMARY_KEYS: set[str] = {"mmlu", "malicious_avg", "malicious_worst", "benign_avg", "benign_worst"}

ROWS_PER_PANEL = 6  # undefended + 5 defenses

# Colormap settings
SR_CMAP_NAME = "magma_r"
SR_CMAP_MIN = 0.02
SR_CMAP_MAX = 0.6

MMLU_CMAP_NAME = "viridis_r"
MMLU_CMAP_MIN = 0.35
MMLU_CMAP_MAX = 0.75

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
    """Format value for cell annotation."""
    return "" if not np.isfinite(value) else f"{value:.2f}"


def get_text_color(value: float, norm: Normalize, cmap: LinearSegmentedColormap) -> str:
    """Return white for dark cells, dark for light cells."""
    if not np.isfinite(value):
        return "#1b1b1b"
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if luminance < 0.5 else "#1b1b1b"


def build_row_vector(
    model_alias: str,
    model_to_idx: dict[str, int],
    attack_to_idx: dict[str, int],
    sr_raw: FloatArray,
    mmlu_raw: FloatArray,
) -> FloatArray:
    """Build a row vector following COLUMN_ORDER for one model."""
    n_cols = len(COLUMN_ORDER)
    row = np.full(n_cols, np.nan)

    if model_alias not in model_to_idx:
        return row

    midx = model_to_idx[model_alias]

    for col_i, col_key in enumerate(COLUMN_ORDER):
        if col_key == "mmlu":
            if BASELINE_ATTACK in attack_to_idx:
                row[col_i] = mmlu_raw[midx, attack_to_idx[BASELINE_ATTACK]]
        elif col_key == "malicious_avg":
            vals = [sr_raw[midx, attack_to_idx[a]] for a in MALICIOUS_ATTACKS if a in attack_to_idx]
            if vals:
                row[col_i] = float(np.nanmean(vals))
        elif col_key == "malicious_worst":
            vals = [sr_raw[midx, attack_to_idx[a]] for a in MALICIOUS_ATTACKS if a in attack_to_idx]
            if vals:
                row[col_i] = float(np.nanmax(vals))
        elif col_key == "benign_avg":
            vals = [sr_raw[midx, attack_to_idx[a]] for a in BENIGN_ATTACKS if a in attack_to_idx]
            if vals:
                row[col_i] = float(np.nanmean(vals))
        elif col_key == "benign_worst":
            vals = [sr_raw[midx, attack_to_idx[a]] for a in BENIGN_ATTACKS if a in attack_to_idx]
            if vals:
                row[col_i] = float(np.nanmax(vals))
        elif col_key in attack_to_idx:
            row[col_i] = sr_raw[midx, attack_to_idx[col_key]]

    return row


# --- Main plotting ---


def plot_heatmap(json_path: Path, output_path: Path) -> None:
    """Plot the defense comparison heatmap."""
    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)

    models: list[str] = data["models"]
    attacks: list[str] = data["attacks"]
    sr_raw: FloatArray = np.array(data["safety_raw"])
    mmlu_raw: FloatArray = np.array(data["utility_raw"])

    model_to_idx = {m: i for i, m in enumerate(models)}
    attack_to_idx = {a: i for i, a in enumerate(attacks)}

    n_cols = len(COLUMN_ORDER)
    n_families = len(MODEL_FAMILIES)
    mmlu_col_idx = COLUMN_ORDER.index("mmlu")

    # Indices of summary columns
    summary_col_indices = {i for i, k in enumerate(COLUMN_ORDER) if k in SUMMARY_KEYS}

    # Colormaps
    sr_cmap = truncated_cmap(SR_CMAP_NAME, SR_CMAP_MIN, SR_CMAP_MAX)
    mmlu_cmap = truncated_cmap(MMLU_CMAP_NAME, MMLU_CMAP_MIN, MMLU_CMAP_MAX)

    # Build all row data to compute norms
    all_rows: list[tuple[str, str, FloatArray]] = []
    for family_name, family_config in MODEL_FAMILIES.items():
        baseline_alias: str = family_config["baseline"]
        row = build_row_vector(baseline_alias, model_to_idx, attack_to_idx, sr_raw, mmlu_raw)
        all_rows.append((family_name, "Undefended", row))
        for def_name, def_alias in family_config["defenses"].items():
            row = build_row_vector(def_alias, model_to_idx, attack_to_idx, sr_raw, mmlu_raw)
            all_rows.append((family_name, def_name, row))

    # SR norm (exclude MMLU column)
    sr_vals = [row[i] for _, _, row in all_rows for i in range(n_cols) if i != mmlu_col_idx]
    sr_finite = [v for v in sr_vals if np.isfinite(v)]
    sr_vmax = max(sr_finite) if sr_finite else 1.0
    sr_norm = Normalize(vmin=0, vmax=sr_vmax)

    # MMLU norm
    mmlu_vals = [row[mmlu_col_idx] for _, _, row in all_rows]
    mmlu_finite = [v for v in mmlu_vals if np.isfinite(v)]
    mmlu_vmin = min(mmlu_finite) if mmlu_finite else 0.0
    mmlu_vmax = max(mmlu_finite) if mmlu_finite else 1.0
    mmlu_norm = Normalize(vmin=mmlu_vmin, vmax=mmlu_vmax)

    # Figure layout — taller rows for more square-like cells
    cell_w = 1.2
    cell_h = 0.55
    fig_width = cell_w * n_cols + 5.0
    fig_height = cell_h * ROWS_PER_PANEL * n_families + 4.0

    fig: Figure = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(n_families, 1, hspace=0.45, top=0.93, bottom=0.14, left=0.09, right=0.82)

    row_cursor = 0
    for panel_idx, (family_name, family_config) in enumerate(MODEL_FAMILIES.items()):
        inner = gridspec.GridSpecFromSubplotSpec(ROWS_PER_PANEL, 1, subplot_spec=outer[panel_idx], hspace=0.0)

        panel_rows = all_rows[row_cursor : row_cursor + ROWS_PER_PANEL]
        row_cursor += ROWS_PER_PANEL

        for row_idx, (_, row_label, row_data) in enumerate(panel_rows):
            ax: Axes = fig.add_subplot(inner[row_idx])

            # Plot SR columns (MMLU masked out)
            sr_row = row_data.copy()
            sr_row[mmlu_col_idx] = np.nan
            ax.imshow(sr_row[np.newaxis, :], cmap=sr_cmap, norm=sr_norm, aspect="auto")

            # Overlay MMLU cell
            mmlu_val = row_data[mmlu_col_idx]
            if np.isfinite(mmlu_val):
                mmlu_arr: FloatArray = np.full((1, n_cols), np.nan)
                mmlu_arr[0, mmlu_col_idx] = mmlu_val
                ax.imshow(mmlu_arr, cmap=mmlu_cmap, norm=mmlu_norm, aspect="auto")

            ax.set_yticks([])
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_xticks(range(n_cols))

            # X-axis labels only on bottom row of bottom panel
            is_last_row = panel_idx == n_families - 1 and row_idx == ROWS_PER_PANEL - 1
            if is_last_row:
                tick_labels = []
                for c in COLUMN_ORDER:
                    tick_labels.append(COLUMN_LABELS.get(c, c))
                ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
                # Bold the summary column tick labels
                for tick_idx, tick_label in enumerate(ax.get_xticklabels()):
                    if tick_idx in summary_col_indices:
                        tick_label.set_fontweight("bold")
            else:
                ax.set_xticklabels([])

            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Draw thick black vertical lines on sides of summary columns
            for col_idx in summary_col_indices:
                for x_off in (-0.5, 0.5):
                    ax.plot(
                        [col_idx + x_off, col_idx + x_off],
                        [-0.5, 0.5],
                        color="black",
                        linewidth=2.0,
                        zorder=5,
                        clip_on=False,
                    )

            # Row label
            is_baseline = row_label == "Undefended"
            ax.set_ylabel(
                row_label,
                rotation=0,
                labelpad=10,
                fontsize=9,
                ha="right",
                va="center",
                fontstyle="italic" if is_baseline else "normal",
            )

            # Cell annotations
            for col_idx in range(n_cols):
                value = row_data[col_idx]
                if not np.isfinite(value):
                    continue
                is_mmlu = col_idx == mmlu_col_idx
                cmap = mmlu_cmap if is_mmlu else sr_cmap
                norm = mmlu_norm if is_mmlu else sr_norm
                ax.text(
                    col_idx,
                    0,
                    format_value(value),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold" if col_idx in summary_col_indices else "normal",
                    color=get_text_color(value, norm, cmap),
                )

            # Panel title on first row
            if row_idx == 0:
                ax.set_title(family_name, fontsize=11, fontweight="bold", pad=10)

    # SR colorbar
    sr_cbar_ax = fig.add_axes((0.84, 0.15, 0.025, 0.70))
    sr_sm = ScalarMappable(norm=sr_norm, cmap=sr_cmap)
    sr_sm.set_array([])
    sr_cbar = plt.colorbar(sr_sm, cax=sr_cbar_ax, orientation="vertical")
    sr_cbar.set_label("StrongReject", rotation=270, labelpad=14, fontsize=9)
    sr_ticks = list(np.linspace(0.0, sr_vmax, 6))
    sr_cbar.set_ticks(sr_ticks)
    sr_cbar.set_ticklabels([f"{t:.2f}" for t in sr_ticks])

    # MMLU colorbar
    mmlu_cbar_ax = fig.add_axes((0.89, 0.15, 0.025, 0.70))
    mmlu_sm = ScalarMappable(norm=mmlu_norm, cmap=mmlu_cmap)
    mmlu_sm.set_array([])
    mmlu_cbar = plt.colorbar(mmlu_sm, cax=mmlu_cbar_ax, orientation="vertical")
    mmlu_cbar.set_label("MMLU-Pro", rotation=270, labelpad=14, fontsize=9)
    mmlu_ticks = list(np.linspace(mmlu_vmin, mmlu_vmax, 6))
    mmlu_cbar.set_ticks(mmlu_ticks)
    mmlu_cbar.set_ticklabels([f"{t:.2f}" for t in mmlu_ticks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Parse arguments and generate defense comparison heatmap."""
    parser = argparse.ArgumentParser(description="Plot defense comparison heatmap")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to heatmap JSON (from scripts/analysis/analyze_results.py)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: same dir as input / heatmap_defenses.png)",
    )
    args = parser.parse_args()

    output: Path = args.output or args.json_path.parent / "heatmap_defenses.png"
    plot_heatmap(args.json_path, output)


if __name__ == "__main__":
    main()
