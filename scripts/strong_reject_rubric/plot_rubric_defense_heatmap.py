#!/usr/bin/env python3
"""Plot defense comparison rubric heatmap — mirrors scripts/figures/plot_defense_heatmap.py.

Produces two styles per metric:
  - Column-based (original): 3 panels, defenses as rows, attacks as columns
  - Row-based (main style): 3 panels, attacks as rows, defenses as columns

Output: rubric_heatmaps/heatmap_rubric_defense_{metric}.png       (column-based)
        rubric_heatmaps/heatmap_rubric_defense_rows_{metric}.png  (row-based)

Usage:
    uv run python scripts/strong_reject_rubric/plot_rubric_defense_heatmap.py \
        results/sweeps/seed_42/aggregated_eps10

    # Single metric only
    uv run python scripts/strong_reject_rubric/plot_rubric_defense_heatmap.py \
        results/sweeps/seed_42/aggregated_eps10 \
        --metric strong_reject_score_rubric

    # Only one style
    uv run python scripts/strong_reject_rubric/plot_rubric_defense_heatmap.py \
        results/sweeps/seed_42/aggregated_eps10 --style columns
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false
# pyright: reportUnusedCallResult=false
from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.gridspec as gridspec
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
                        ("TAR", "llama3_8b_tar_o"),
                        ("T-Vaccine", "llama3_8b_t_vaccine"),
                        ("SDD", "llama3_8b_sdd"),
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
                        ("TAR", "llama3_8b_instruct_tar_o"),
                        ("T-Vaccine", "llama3_8b_instruct_t_vaccine"),
                        ("SDD", "llama3_8b_instruct_sdd"),
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
                        ("TAR", "qwen3_8b_tar_o"),
                        ("T-Vaccine", "qwen3_8b_t_vaccine"),
                        ("SDD", "qwen3_8b_sdd"),
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

# Column order for column-based style
COLUMN_ORDER: list[str] = (
    [BASELINE_ATTACK]
    + MALICIOUS_ATTACKS
    + ["malicious_avg", "malicious_worst"]
    + BENIGN_ATTACKS
    + ["benign_avg", "benign_worst"]
    + [EMBEDDING_ATTACK]
)

COLUMN_LABELS: dict[str, str] = {
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

SUMMARY_KEYS: set[str] = {"malicious_avg", "malicious_worst", "benign_avg", "benign_worst"}

ROWS_PER_PANEL = 8  # undefended + 7 defenses

# --- Metric configurations ---

METRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "strong_reject_score_gemma": {
        "display_name": "SR Score (Gemma)",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.6),
        "source_dir": "strong_reject_rubric",
    },
    "strong_reject_score_rubric": {
        "display_name": "SR Score (Rubric)",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "magma_r",
        "cmap_range": (0.02, 0.6),
        "source_dir": "strong_reject_rubric",
    },
    "refusal": {
        "display_name": "Refusal Rate",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Blues",
        "cmap_range": (0.02, 0.6),
        "source_dir": "strong_reject_rubric",
    },
    "convincingness": {
        "display_name": "Convincingness (Non-Refused)",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Purples",
        "cmap_range": (0.02, 0.6),
        "source_dir": "strong_reject_rubric",
    },
    "specificity": {
        "display_name": "Specificity (Non-Refused)",
        "vmin": 0.0,
        "vmax": 5.0,
        "cmap": "Oranges",
        "cmap_range": (0.02, 0.6),
        "source_dir": "strong_reject_rubric",
    },
}

ALL_METRICS: list[str] = list(METRIC_CONFIGS.keys())

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


def display_attack_name(attack: str) -> str:
    return attack.replace("_", " ").title()


# --- Data collection ---


def collect_scores_flat(
    agg_dir: Path,
    metric_name: str,
) -> dict[tuple[str, str], float]:
    """Collect scores as {(model_alias, attack_name): value}."""
    source_dir = METRIC_CONFIGS[metric_name]["source_dir"]
    scores: dict[tuple[str, str], float] = {}

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
                        break
            except Exception as e:
                LOGGER.warning(f"Error reading {model_dir.name}/{attack_dir.name}: {e}")

    return scores


def build_column_row_vector(
    model_alias: str,
    scores: dict[tuple[str, str], float],
) -> FloatArray:
    """Build a row vector following COLUMN_ORDER for one model (column-based style)."""
    n_cols = len(COLUMN_ORDER)
    row = np.full(n_cols, np.nan)

    for col_i, col_key in enumerate(COLUMN_ORDER):
        if col_key == "malicious_avg":
            vals = [scores[(model_alias, a)] for a in MALICIOUS_ATTACKS if (model_alias, a) in scores]
            if vals:
                row[col_i] = float(np.nanmean(vals))
        elif col_key == "malicious_worst":
            vals = [scores[(model_alias, a)] for a in MALICIOUS_ATTACKS if (model_alias, a) in scores]
            if vals:
                row[col_i] = float(np.nanmax(vals))
        elif col_key == "benign_avg":
            vals = [scores[(model_alias, a)] for a in BENIGN_ATTACKS if (model_alias, a) in scores]
            if vals:
                row[col_i] = float(np.nanmean(vals))
        elif col_key == "benign_worst":
            vals = [scores[(model_alias, a)] for a in BENIGN_ATTACKS if (model_alias, a) in scores]
            if vals:
                row[col_i] = float(np.nanmax(vals))
        elif (model_alias, col_key) in scores:
            row[col_i] = scores[(model_alias, col_key)]

    return row


# --- Column-based plotting (defenses as rows, attacks as columns) ---


def plot_defense_heatmap_columns(
    agg_dir: Path,
    metric_name: str,
    output_path: Path,
) -> None:
    """Plot the column-based defense comparison heatmap."""
    config = METRIC_CONFIGS[metric_name]
    scores = collect_scores_flat(agg_dir, metric_name)

    if not scores:
        LOGGER.warning(f"No scores found for {metric_name}, skipping")
        return

    n_cols = len(COLUMN_ORDER)
    n_families = len(MODEL_FAMILIES)
    summary_col_indices = {i for i, k in enumerate(COLUMN_ORDER) if k in SUMMARY_KEYS}

    cmap = truncated_cmap(config["cmap"], config["cmap_range"][0], config["cmap_range"][1])

    # Build all row data
    all_rows: list[tuple[str, str, FloatArray]] = []
    for family_name, family_config in MODEL_FAMILIES.items():
        row = build_column_row_vector(family_config["baseline"], scores)
        all_rows.append((family_name, "Undefended", row))
        for def_name, def_alias in family_config["defenses"].items():
            row = build_column_row_vector(def_alias, scores)
            all_rows.append((family_name, def_name, row))

    # Compute norm
    all_vals = [row[i] for _, _, row in all_rows for i in range(n_cols)]
    finite_vals = [v for v in all_vals if np.isfinite(v)]
    data_vmax = max(finite_vals) if finite_vals else config["vmax"]
    norm = Normalize(vmin=config["vmin"], vmax=min(data_vmax, config["vmax"]))

    # Figure layout
    cell_w = 1.2
    cell_h = 0.55
    fig_width = cell_w * n_cols + 5.0
    fig_height = cell_h * ROWS_PER_PANEL * n_families + 4.0

    fig: Figure = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(config["display_name"], fontsize=14, fontweight="bold", y=0.97)
    outer = gridspec.GridSpec(n_families, 1, hspace=0.45, top=0.93, bottom=0.14, left=0.09, right=0.82)

    row_cursor = 0
    for panel_idx, (family_name, _family_config) in enumerate(MODEL_FAMILIES.items()):
        inner = gridspec.GridSpecFromSubplotSpec(ROWS_PER_PANEL, 1, subplot_spec=outer[panel_idx], hspace=0.0)

        panel_rows = all_rows[row_cursor : row_cursor + ROWS_PER_PANEL]
        row_cursor += ROWS_PER_PANEL

        for row_idx, (_, row_label, row_data) in enumerate(panel_rows):
            ax: Axes = fig.add_subplot(inner[row_idx])
            ax.imshow(row_data[np.newaxis, :], cmap=cmap, norm=norm, aspect="auto")

            ax.set_yticks([])
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_xticks(range(n_cols))

            is_last_row = panel_idx == n_families - 1 and row_idx == ROWS_PER_PANEL - 1
            if is_last_row:
                tick_labels = [COLUMN_LABELS.get(c, c) for c in COLUMN_ORDER]
                ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
                for tick_idx, tick_label in enumerate(ax.get_xticklabels()):
                    if tick_idx in summary_col_indices:
                        tick_label.set_fontweight("bold")
            else:
                ax.set_xticklabels([])

            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

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

            for col_idx in range(n_cols):
                value = row_data[col_idx]
                if not np.isfinite(value):
                    continue
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

            if row_idx == 0:
                ax.set_title(family_name, fontsize=11, fontweight="bold", pad=10)

    # Colorbar
    cbar_ax = fig.add_axes((0.84, 0.15, 0.025, 0.70))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(config["display_name"], rotation=270, labelpad=14, fontsize=9)
    ticks = list(np.linspace(config["vmin"], norm.vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path}")


# --- Row-based plotting (attacks as rows, defenses as columns) ---


MMLU_CMAP_NAME = "viridis_r"
MMLU_CMAP_MIN = 0.35
MMLU_CMAP_MAX = 0.75


def _load_mmlu_scores(agg_dir: Path, model_aliases: list[str]) -> FloatArray:
    """Load MMLU-Pro scores for untampered baseline.

    Reads from mmlu_pro_val/evaluation.json in each model's no_weight_modification eval dir.
    Falls back to heatmap_max_sr.json for any models not found in the eval dirs.
    """
    mmlu_row = np.full(len(model_aliases), np.nan)

    # Try reading directly from eval dirs first
    for col_idx, alias in enumerate(model_aliases):
        baseline_dir = agg_dir / alias / BASELINE_ATTACK
        if not baseline_dir.exists():
            continue
        eval_dirs = list(baseline_dir.glob("trial_*_tamperbench_evaluation"))
        if not eval_dirs:
            continue
        mmlu_file = eval_dirs[0] / "mmlu_pro_val" / "evaluation.json"
        if mmlu_file.exists():
            try:
                with open(mmlu_file) as f:
                    metrics = json.load(f)
                for m in metrics:
                    if m["metric_name"] == "mmlu_pro_accuracy":
                        mmlu_row[col_idx] = m["metric_value"]
                        break
            except Exception:
                pass

    # Fall back to heatmap JSON for any remaining NaNs
    json_path = agg_dir / "heatmap_max_sr.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        models_json: list[str] = data["models"]
        attacks_json: list[str] = data["attacks"]
        utility_raw: FloatArray = np.array(data["utility_raw"])
        model_to_idx = {m: i for i, m in enumerate(models_json)}
        baseline_idx = attacks_json.index(BASELINE_ATTACK) if BASELINE_ATTACK in attacks_json else None

        if baseline_idx is not None:
            for col_idx, alias in enumerate(model_aliases):
                if np.isnan(mmlu_row[col_idx]) and alias in model_to_idx:
                    mmlu_row[col_idx] = utility_raw[model_to_idx[alias], baseline_idx]

    return mmlu_row


def plot_defense_heatmap_rows(
    agg_dir: Path,
    metric_name: str,
    output_path: Path,
) -> None:
    """Plot row-based defense heatmap: attacks as rows, all defense models as columns (single heatmap)."""
    config = METRIC_CONFIGS[metric_name]
    scores = collect_scores_flat(agg_dir, metric_name)

    if not scores:
        LOGGER.warning(f"No scores found for {metric_name}, skipping")
        return

    sr_cmap = truncated_cmap(config["cmap"], config["cmap_range"][0], config["cmap_range"][1])
    sr_norm = Normalize(vmin=config["vmin"], vmax=config["vmax"])

    mmlu_cmap = truncated_cmap(MMLU_CMAP_NAME, MMLU_CMAP_MIN, MMLU_CMAP_MAX)

    # Build flat column list: all models across all families
    model_aliases: list[str] = []
    col_labels: list[str] = []
    for family_name, family_config in MODEL_FAMILIES.items():
        short_family = family_name
        model_aliases.append(family_config["baseline"])
        col_labels.append(f"{short_family}\nUndefended")
        for def_name, def_alias in family_config["defenses"].items():
            model_aliases.append(def_alias)
            col_labels.append(f"{short_family}\n{def_name}")
    n_models = len(model_aliases)

    # Load MMLU scores
    mmlu_row = _load_mmlu_scores(agg_dir, model_aliases)
    mmlu_finite = mmlu_row[np.isfinite(mmlu_row)]
    mmlu_vmin = float(np.min(mmlu_finite)) if len(mmlu_finite) > 0 else 0.0
    mmlu_vmax = float(np.max(mmlu_finite)) if len(mmlu_finite) > 0 else 1.0
    mmlu_norm = Normalize(vmin=mmlu_vmin, vmax=mmlu_vmax)

    def get_score(model_alias: str, attack_key: str) -> float:
        if attack_key == "malicious_avg":
            vals = [scores[(model_alias, a)] for a in MALICIOUS_ATTACKS if (model_alias, a) in scores]
            return float(np.nanmean(vals)) if vals else float("nan")
        elif attack_key == "benign_avg":
            vals = [scores[(model_alias, a)] for a in BENIGN_ATTACKS if (model_alias, a) in scores]
            return float(np.nanmean(vals)) if vals else float("nan")
        else:
            return scores.get((model_alias, attack_key), float("nan"))

    # Build row order: MMLU first, then baseline SR, then attacks with averages
    # row_type: "mmlu" or "sr"
    row_entries: list[tuple[str, str, str]] = []  # (label, attack_key, row_type)

    row_entries.append(("MMLU-Pro (untampered)", "__mmlu__", "mmlu"))
    row_entries.append((display_attack_name(BASELINE_ATTACK), BASELINE_ATTACK, "sr"))

    for a in MALICIOUS_ATTACKS:
        row_entries.append((display_attack_name(a), a, "sr"))
    row_entries.append(("Avg: Malicious", "malicious_avg", "sr"))

    for a in BENIGN_ATTACKS:
        row_entries.append((display_attack_name(a), a, "sr"))
    row_entries.append(("Avg: Benign", "benign_avg", "sr"))

    row_entries.append((display_attack_name(EMBEDDING_ATTACK), EMBEDDING_ATTACK, "sr"))

    # Overall Max (across all non-baseline attacks)
    row_entries.append(("Overall Max", "__overall_max__", "sr"))

    n_rows = len(row_entries)

    # All non-baseline attacks for computing overall max
    all_attacks = MALICIOUS_ATTACKS + BENIGN_ATTACKS + [EMBEDDING_ATTACK]

    # Build matrix
    matrix = np.full((n_rows, n_models), np.nan)
    labels: list[str] = []
    row_types: list[str] = []
    for row_idx, (row_label, attack_key, row_type) in enumerate(row_entries):
        labels.append(row_label)
        row_types.append(row_type)
        if row_type == "mmlu":
            matrix[row_idx, :] = mmlu_row
        elif attack_key == "__overall_max__":
            for col_idx, model_alias in enumerate(model_aliases):
                vals = [get_score(model_alias, a) for a in all_attacks]
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    matrix[row_idx, col_idx] = max(finite)
        else:
            for col_idx, model_alias in enumerate(model_aliases):
                matrix[row_idx, col_idx] = get_score(model_alias, attack_key)

    # Figure layout
    fig_width = max(18.0, 0.9 * n_models)
    fig_height = max(5.4, 0.54 * n_rows)

    fig: Figure
    axes: npt.NDArray[Any]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.93, bottom=0.12, hspace=0.0)
    fig.suptitle(config["display_name"], fontsize=14, fontweight="bold", y=0.96)

    # Family boundary columns (for vertical divider lines)
    family_boundaries: list[int] = []
    col_cursor = 0
    for family_config in MODEL_FAMILIES.values():
        n_in_family = 1 + len(family_config["defenses"])
        col_cursor += n_in_family
        family_boundaries.append(col_cursor)
    family_boundaries.pop()  # no line after the last family

    for row_idx, ax in enumerate(axes):
        ax: Axes
        row_data = matrix[row_idx, :][np.newaxis, :]
        is_mmlu = row_types[row_idx] == "mmlu"

        if is_mmlu:
            cur_cmap, cur_norm = mmlu_cmap, mmlu_norm
        else:
            cur_cmap, cur_norm = sr_cmap, sr_norm

        ax.imshow(row_data, cmap=cur_cmap, norm=cur_norm, aspect="auto")

        ax.set_yticks([])
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_xticks(range(n_models))

        if row_idx == n_rows - 1:
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Vertical dashed lines between model families
        for boundary_col in family_boundaries:
            ax.axvline(x=boundary_col - 0.5, color="#555555", linewidth=1.5, linestyle="--", zorder=5, clip_on=False)

        label = labels[row_idx]
        is_avg = label.startswith("Avg:") or label.startswith("Overall")
        ax.set_ylabel(
            label,
            rotation=0,
            labelpad=18,
            fontsize=8,
            ha="right",
            va="center",
            fontweight="bold" if is_avg else "normal",
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
                    fontweight="bold" if is_avg else "normal",
                    color=get_text_color(value, cur_norm, cur_cmap),
                )

    # SR Colorbar
    sr_cbar_ax = fig.add_axes((0.84, 0.15, 0.025, 0.70))
    sr_sm = ScalarMappable(norm=sr_norm, cmap=sr_cmap)
    sr_sm.set_array([])
    sr_cbar = plt.colorbar(sr_sm, cax=sr_cbar_ax, orientation="vertical")
    sr_cbar.set_label(config["display_name"], rotation=270, labelpad=14, fontsize=9)
    sr_ticks = list(np.linspace(config["vmin"], config["vmax"], 6))
    sr_cbar.set_ticks(sr_ticks)
    sr_cbar.set_ticklabels([f"{t:.2f}" for t in sr_ticks])

    # MMLU Colorbar
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
    LOGGER.info(f"Saved: {output_path}")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot defense comparison rubric heatmaps",
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
        "--style",
        choices=["columns", "rows", "both"],
        default="both",
        help="Which style to generate (default: both)",
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
        LOGGER.info(f"Processing: {METRIC_CONFIGS[metric_name]['display_name']}")

        if args.style in ("columns", "both"):
            output_path = output_dir / f"heatmap_rubric_defense_{metric_name}.png"
            plot_defense_heatmap_columns(args.agg_dir, metric_name, output_path)

        if args.style in ("rows", "both"):
            output_path = output_dir / f"heatmap_rubric_defense_rows_{metric_name}.png"
            plot_defense_heatmap_rows(args.agg_dir, metric_name, output_path)

    LOGGER.info(f"Done! Heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
