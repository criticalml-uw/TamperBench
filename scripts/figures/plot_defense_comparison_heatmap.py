#!/usr/bin/env python3
"""Plot defense comparison heatmap: tuned vs non-tuned defenses under weak/strong attacks.

Produces 3 versions:
  1. Raw: all absolute values
  2. Delta: MMLU rows shown as delta (from undefended for untampered, from untampered for post-attack)
  3. Combined: raw + delta MMLU rows together

SR rows are always raw (never deltas).

Rows for raw/delta:
  1. MMLU-Pro (untampered)         [raw or delta from undefended]
  2. Pre-Attack SR
  3. Post-Attack SR (Strong)
  4. MMLU-Pro post Strong           [raw or delta from untampered]
  5. Post-Attack SR (Weak)
  6. MMLU-Pro post Weak             [raw or delta from untampered]

Usage:
    uv run python scripts/figures/plot_defense_comparison_heatmap.py \
        --strong /data/far_ai_group/saad_ws/results/sweeps/seed_42/aggregated_eps10 \
        --weak /data/far_ai_group/saad_ws/results/targeted_sweeps_2/seed_42/aggregated_eps10

    # Exclude CRL
    uv run python scripts/figures/plot_defense_comparison_heatmap.py \
        --strong .../aggregated_eps10 --weak .../aggregated_eps10 --exclude CRL
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

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# --- Model families ---

MODEL_FAMILIES: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "Llama-3-8B",
            {
                "baseline": "llama3_8b_baseline",
                "models": OrderedDict(
                    [
                        ("Undefended", "llama3_8b_baseline"),
                        ("Booster\n(non-tuned)", "llama3_8b_booster_non_tuned"),
                        ("Booster\n(tuned)", "llama3_8b_booster"),
                        ("CRL\n(non-tuned)", "llama3_8b_crl_non_tuned"),
                        ("CRL\n(tuned)", "llama3_8b_crl"),
                        ("TAR\n(non-tuned)", "llama3_8b_tar_v2_non_tuned"),
                        ("TAR\n(tuned)", "llama3_8b_tar"),
                    ]
                ),
            },
        ),
        (
            "Llama-3-8B-Instruct",
            {
                "baseline": "llama3_8b_instruct_baseline",
                "models": OrderedDict(
                    [
                        ("Undefended", "llama3_8b_instruct_baseline"),
                        ("Booster\n(non-tuned)", "llama3_8b_instruct_booster_non_tuned"),
                        ("Booster\n(tuned)", "llama3_8b_instruct_booster"),
                        ("CRL\n(non-tuned)", "llama3_8b_instruct_crl_non_tuned"),
                        ("CRL\n(tuned)", "llama3_8b_instruct_crl"),
                        ("TAR\n(non-tuned)", "llama3_8b_instruct_tar_v2_non_tuned"),
                        ("TAR\n(tuned)", "llama3_8b_instruct_tar"),
                    ]
                ),
            },
        ),
        (
            "Qwen3-8B",
            {
                "baseline": "qwen3_8b",
                "models": OrderedDict(
                    [
                        ("Undefended", "qwen3_8b"),
                        ("Booster\n(non-tuned)", "qwen3_8b_booster_non_tuned"),
                        ("Booster\n(tuned)", "qwen3_8b_booster"),
                        ("CRL\n(non-tuned)", "qwen3_8b_crl_non_tuned"),
                        ("CRL\n(tuned)", "qwen3_8b_crl"),
                        ("TAR\n(non-tuned)", "qwen3_8b_tar_v2_non_tuned"),
                        ("TAR\n(tuned)", "qwen3_8b_tar"),
                    ]
                ),
            },
        ),
    ]
)

# --- Colormap settings ---

SR_CMAP_NAME = "magma_r"
SR_CMAP_MIN = 0.02
SR_CMAP_MAX = 0.75

MMLU_CMAP_NAME = "viridis_r"
MMLU_CMAP_MIN = 0.35
MMLU_CMAP_MAX = 0.75

FloatArray = npt.NDArray[np.floating[Any]]


# --- Helpers ---


def truncated_cmap(name: str, minval: float, maxval: float) -> LinearSegmentedColormap:
    base = plt.get_cmap(name)
    colors = base(np.linspace(minval, maxval, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_trunc", colors)
    cmap.set_bad((0.96, 0.96, 0.96, 0.0))
    return cmap


def format_value(value: float, is_delta: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:+.2f}" if is_delta else f"{value:.2f}"


def get_text_color(
    value: float, norm: Normalize | TwoSlopeNorm, cmap: LinearSegmentedColormap
) -> str:
    if not np.isfinite(value):
        return "#1b1b1b"
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if luminance < 0.5 else "#1b1b1b"


# --- Data loading ---


def _read_rubric_metric(
    agg_dir: Path, model_alias: str, attack: str, metric_name: str
) -> float:
    model_dir = agg_dir / model_alias / attack
    if not model_dir.exists():
        return float("nan")
    eval_dirs = list(model_dir.glob("trial_*_tamperbench_evaluation"))
    if not eval_dirs:
        return float("nan")
    rubric_file = eval_dirs[0] / "strong_reject_rubric" / "evaluation.json"
    if not rubric_file.exists():
        return float("nan")
    try:
        with open(rubric_file) as f:
            metrics = json.load(f)
        for m in metrics:
            if m["metric_name"] == metric_name:
                val = m["metric_value"]
                return val if val is not None and np.isfinite(val) else float("nan")
    except Exception:
        pass
    return float("nan")


def _read_mmlu(agg_dir: Path, model_alias: str, attack: str = "no_weight_modification") -> float:
    model_dir = agg_dir / model_alias / attack
    if not model_dir.exists():
        return float("nan")
    eval_dirs = list(model_dir.glob("trial_*_tamperbench_evaluation"))
    if not eval_dirs:
        return float("nan")
    mmlu_file = eval_dirs[0] / "mmlu_pro_val" / "evaluation.json"
    if not mmlu_file.exists():
        return float("nan")
    try:
        with open(mmlu_file) as f:
            metrics = json.load(f)
        for m in metrics:
            if m["metric_name"] == "mmlu_pro_accuracy":
                return m["metric_value"]
    except Exception:
        pass
    return float("nan")


# --- Data collection ---


def collect_all_data(
    strong_dir: Path,
    weak_dir: Path,
    model_aliases: list[str],
    metric: str,
) -> dict[str, FloatArray]:
    n = len(model_aliases)
    data: dict[str, FloatArray] = {
        "mmlu_untampered": np.full(n, np.nan),
        "pre_atk_sr": np.full(n, np.nan),
        "strong_sr": np.full(n, np.nan),
        "strong_mmlu": np.full(n, np.nan),
        "weak_sr": np.full(n, np.nan),
        "weak_mmlu": np.full(n, np.nan),
    }
    for i, alias in enumerate(model_aliases):
        data["mmlu_untampered"][i] = _read_mmlu(strong_dir, alias, "no_weight_modification")
        data["pre_atk_sr"][i] = _read_rubric_metric(
            strong_dir, alias, "no_weight_modification", metric
        )
        data["strong_sr"][i] = _read_rubric_metric(strong_dir, alias, "lora_finetune", metric)
        data["strong_mmlu"][i] = _read_mmlu(strong_dir, alias, "lora_finetune")
        data["weak_sr"][i] = _read_rubric_metric(weak_dir, alias, "lora_finetune", metric)
        data["weak_mmlu"][i] = _read_mmlu(weak_dir, alias, "lora_finetune")
    return data


def get_baseline_idx_map(
    model_aliases: list[str],
    exclude: set[str],
) -> dict[int, int]:
    """Map each column index to its family's undefended baseline column index."""
    alias_to_idx = {a: i for i, a in enumerate(model_aliases)}
    mapping: dict[int, int] = {}
    for family_config in MODEL_FAMILIES.values():
        baseline_alias = family_config["baseline"]
        baseline_col = alias_to_idx.get(baseline_alias)
        for def_label, alias in family_config["models"].items():
            defense_name = def_label.split("\n")[0]
            if defense_name in exclude:
                continue
            if alias in alias_to_idx and baseline_col is not None:
                mapping[alias_to_idx[alias]] = baseline_col
    return mapping


# --- Plotting ---


def _plot_heatmap(
    matrix: FloatArray,
    labels: list[str],
    row_is_mmlu: list[bool],
    row_is_delta: list[bool],
    col_labels: list[str],
    family_boundaries: list[int],
    n_models: int,
    title: str,
    output_path: Path,
    sr_label: str = "StrongReject",
) -> None:
    n_rows = len(labels)

    sr_cmap = truncated_cmap(SR_CMAP_NAME, SR_CMAP_MIN, SR_CMAP_MAX)
    sr_norm = Normalize(vmin=0, vmax=1.0)

    mmlu_cmap = truncated_cmap(MMLU_CMAP_NAME, MMLU_CMAP_MIN, MMLU_CMAP_MAX)

    # MMLU raw norm
    mmlu_raw_vals = []
    for row_idx in range(n_rows):
        if row_is_mmlu[row_idx] and not row_is_delta[row_idx]:
            mmlu_raw_vals.extend(matrix[row_idx][np.isfinite(matrix[row_idx])].tolist())
    mmlu_vmin = min(mmlu_raw_vals) if mmlu_raw_vals else 0.0
    mmlu_vmax = max(mmlu_raw_vals) if mmlu_raw_vals else 1.0
    mmlu_norm = Normalize(vmin=mmlu_vmin, vmax=mmlu_vmax)

    # MMLU delta norm (use same viridis cmap but with TwoSlopeNorm)
    mmlu_delta_vals = []
    for row_idx in range(n_rows):
        if row_is_mmlu[row_idx] and row_is_delta[row_idx]:
            mmlu_delta_vals.extend(matrix[row_idx][np.isfinite(matrix[row_idx])].tolist())
    mmlu_delta_abs = max(abs(v) for v in mmlu_delta_vals) if mmlu_delta_vals else 0.1
    mmlu_delta_norm = TwoSlopeNorm(vmin=-mmlu_delta_abs, vcenter=0.0, vmax=mmlu_delta_abs)

    fig_width = max(18.0, 0.9 * n_models)
    fig_height = max(4.0, 0.65 * n_rows)

    fig: Figure
    axes: npt.NDArray[Any]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.18, hspace=0.0)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    for row_idx, ax in enumerate(axes):
        ax: Axes
        row_data = matrix[row_idx, :][np.newaxis, :]
        is_mmlu = row_is_mmlu[row_idx]
        is_delta = row_is_delta[row_idx]

        if is_mmlu and is_delta:
            cur_cmap, cur_norm = mmlu_cmap, mmlu_delta_norm
        elif is_mmlu:
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

        for boundary_col in family_boundaries:
            ax.axvline(
                x=boundary_col - 0.5, color="#555555", linewidth=1.5,
                linestyle="--", zorder=5, clip_on=False,
            )

        label = labels[row_idx]
        is_bold = "Post-Attack" in label
        ax.set_ylabel(
            label, rotation=0, labelpad=18, fontsize=8,
            ha="right", va="center",
            fontweight="bold" if is_bold else "normal",
        )

        for col_idx in range(n_models):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx, 0, format_value(value, is_delta),
                    ha="center", va="center", fontsize=8,
                    color=get_text_color(value, cur_norm, cur_cmap),
                )

    # SR Colorbar
    sr_cbar_ax = fig.add_axes((0.84, 0.20, 0.025, 0.65))
    sr_sm = ScalarMappable(norm=sr_norm, cmap=sr_cmap)
    sr_sm.set_array([])
    sr_cbar = plt.colorbar(sr_sm, cax=sr_cbar_ax, orientation="vertical")
    sr_cbar.set_label(sr_label, rotation=270, labelpad=14, fontsize=9)
    sr_ticks = list(np.linspace(0, 1.0, 6))
    sr_cbar.set_ticks(sr_ticks)
    sr_cbar.set_ticklabels([f"{t:.2f}" for t in sr_ticks])

    # MMLU Colorbar — use delta norm if any MMLU rows are deltas
    has_mmlu_deltas = any(row_is_mmlu[i] and row_is_delta[i] for i in range(n_rows))
    has_mmlu_raw = any(row_is_mmlu[i] and not row_is_delta[i] for i in range(n_rows))

    if has_mmlu_deltas and not has_mmlu_raw:
        # All MMLU rows are deltas
        mmlu_cbar_ax = fig.add_axes((0.89, 0.20, 0.025, 0.65))
        mmlu_sm = ScalarMappable(norm=mmlu_delta_norm, cmap=mmlu_cmap)
        mmlu_sm.set_array([])
        mmlu_cbar = plt.colorbar(mmlu_sm, cax=mmlu_cbar_ax, orientation="vertical")
        mmlu_cbar.set_label("MMLU-Pro \u0394", rotation=270, labelpad=14, fontsize=9)
        mmlu_ticks = list(np.linspace(-mmlu_delta_abs, mmlu_delta_abs, 7))
        mmlu_cbar.set_ticks(mmlu_ticks)
        mmlu_cbar.set_ticklabels([f"{t:+.2f}" for t in mmlu_ticks])
    else:
        # Raw or mixed — show raw colorbar
        mmlu_cbar_ax = fig.add_axes((0.89, 0.20, 0.025, 0.65))
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


# --- Version builders ---


def build_raw(data: dict[str, FloatArray]) -> tuple[FloatArray, list[str], list[bool], list[bool]]:
    """Raw absolute values."""
    rows = [
        data["mmlu_untampered"],
        data["pre_atk_sr"],
        data["strong_sr"],
        data["strong_mmlu"],
        data["weak_sr"],
        data["weak_mmlu"],
    ]
    labels = [
        "MMLU-Pro (untampered)",
        "Pre-Attack SR",
        "Post-Attack SR (Strong)",
        "MMLU-Pro post Strong",
        "Post-Attack SR (Weak)",
        "MMLU-Pro post Weak",
    ]
    is_mmlu = [True, False, False, True, False, True]
    is_delta = [False] * 6
    return np.stack(rows), labels, is_mmlu, is_delta


def build_delta(
    data: dict[str, FloatArray],
    baseline_idx_map: dict[int, int],
    n_models: int,
) -> tuple[FloatArray, list[str], list[bool], list[bool]]:
    """MMLU as deltas, SR stays raw."""
    # MMLU delta from undefended baseline (per family)
    mmlu_delta_untampered = np.full(n_models, np.nan)
    for col_idx in range(n_models):
        if col_idx in baseline_idx_map:
            base_idx = baseline_idx_map[col_idx]
            base_val = data["mmlu_untampered"][base_idx]
            cur_val = data["mmlu_untampered"][col_idx]
            if np.isfinite(base_val) and np.isfinite(cur_val):
                mmlu_delta_untampered[col_idx] = cur_val - base_val

    # Post-attack MMLU delta from untampered (per model)
    mmlu_delta_strong = data["strong_mmlu"] - data["mmlu_untampered"]
    mmlu_delta_weak = data["weak_mmlu"] - data["mmlu_untampered"]

    rows = [
        data["pre_atk_sr"],
        mmlu_delta_untampered,
        data["strong_sr"],
        mmlu_delta_strong,
        data["weak_sr"],
        mmlu_delta_weak,
    ]
    labels = [
        "Pre-Attack SR",
        "\u0394 MMLU (from undefended)",
        "Post-Attack SR (Strong)",
        "\u0394 MMLU post Strong",
        "Post-Attack SR (Weak)",
        "\u0394 MMLU post Weak",
    ]
    is_mmlu = [False, True, False, True, False, True]
    is_delta = [False, True, False, True, False, True]
    return np.stack(rows), labels, is_mmlu, is_delta


def build_combined(
    data: dict[str, FloatArray],
    baseline_idx_map: dict[int, int],
    n_models: int,
) -> tuple[FloatArray, list[str], list[bool], list[bool]]:
    """Raw + delta MMLU rows together, SR always raw."""
    mmlu_delta_untampered = np.full(n_models, np.nan)
    for col_idx in range(n_models):
        if col_idx in baseline_idx_map:
            base_idx = baseline_idx_map[col_idx]
            base_val = data["mmlu_untampered"][base_idx]
            cur_val = data["mmlu_untampered"][col_idx]
            if np.isfinite(base_val) and np.isfinite(cur_val):
                mmlu_delta_untampered[col_idx] = cur_val - base_val

    mmlu_delta_strong = data["strong_mmlu"] - data["mmlu_untampered"]
    mmlu_delta_weak = data["weak_mmlu"] - data["mmlu_untampered"]

    rows = [
        data["mmlu_untampered"],
        mmlu_delta_untampered,
        data["pre_atk_sr"],
        data["strong_sr"],
        data["strong_mmlu"],
        mmlu_delta_strong,
        data["weak_sr"],
        data["weak_mmlu"],
        mmlu_delta_weak,
    ]
    labels = [
        "MMLU-Pro (untampered)",
        "\u0394 MMLU (from undefended)",
        "Pre-Attack SR",
        "Post-Attack SR (Strong)",
        "MMLU-Pro post Strong",
        "\u0394 MMLU post Strong",
        "Post-Attack SR (Weak)",
        "MMLU-Pro post Weak",
        "\u0394 MMLU post Weak",
    ]
    is_mmlu = [True, True, False, False, True, True, False, True, True]
    is_delta = [False, True, False, False, False, True, False, False, True]
    return np.stack(rows), labels, is_mmlu, is_delta


# --- Main ---


def plot_defense_comparison(
    strong_dir: Path,
    weak_dir: Path,
    output_dir: Path,
    metric: str = "strong_reject_score_rubric",
    exclude_defenses: list[str] | None = None,
) -> None:
    exclude = set(exclude_defenses or [])

    model_aliases: list[str] = []
    col_labels: list[str] = []
    family_boundaries: list[int] = []
    col_cursor = 0

    for family_name, family_config in MODEL_FAMILIES.items():
        count = 0
        for def_label, alias in family_config["models"].items():
            defense_name = def_label.split("\n")[0]
            if defense_name in exclude:
                continue
            model_aliases.append(alias)
            col_labels.append(f"{family_name}\n{def_label}")
            count += 1
        col_cursor += count
        family_boundaries.append(col_cursor)
    family_boundaries.pop()

    n_models = len(model_aliases)
    data = collect_all_data(strong_dir, weak_dir, model_aliases, metric)
    baseline_idx_map = get_baseline_idx_map(model_aliases, exclude)

    sr_label = "StrongReject (Rubric)" if "rubric" in metric else "StrongReject (Gemma)"
    suffix = "" if exclude_defenses is None else "_no_" + "_".join(exclude_defenses).lower()

    # Version 1: Raw
    matrix, labels, is_mmlu, is_delta = build_raw(data)
    _plot_heatmap(
        matrix, labels, is_mmlu, is_delta, col_labels, family_boundaries,
        n_models, "Defense Comparison: Raw Scores", output_dir / f"defense_comparison_raw{suffix}.png",
        sr_label=sr_label,
    )

    # Version 2: Delta (MMLU only, SR raw)
    matrix, labels, is_mmlu, is_delta = build_delta(data, baseline_idx_map, n_models)
    _plot_heatmap(
        matrix, labels, is_mmlu, is_delta, col_labels, family_boundaries,
        n_models, "Defense Comparison: MMLU Deltas", output_dir / f"defense_comparison_delta{suffix}.png",
        sr_label=sr_label,
    )

    # Version 3: Combined
    matrix, labels, is_mmlu, is_delta = build_combined(data, baseline_idx_map, n_models)
    _plot_heatmap(
        matrix, labels, is_mmlu, is_delta, col_labels, family_boundaries,
        n_models, "Defense Comparison: Combined", output_dir / f"defense_comparison_combined{suffix}.png",
        sr_label=sr_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot defense comparison heatmaps (tuned vs non-tuned, 3 versions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strong", type=Path, required=True, help="Aggregated dir for strong attack")
    parser.add_argument("--weak", type=Path, required=True, help="Aggregated dir for weak attack")
    parser.add_argument("--metric", default="strong_reject_score_rubric", help="Metric (default: strong_reject_score_rubric)")
    parser.add_argument("--exclude", nargs="+", default=None, help="Defense names to exclude (e.g., --exclude CRL)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory (default: strong dir)")
    args = parser.parse_args()

    output_dir: Path = args.output_dir or args.strong
    plot_defense_comparison(
        args.strong, args.weak, output_dir,
        metric=args.metric, exclude_defenses=args.exclude,
    )


if __name__ == "__main__":
    main()
