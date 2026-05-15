#!/usr/bin/env python3
"""Generate strong_reject_rubric_plus_plus from existing strong_reject_rubric directories.

This script reads existing strong_reject_rubric evaluations and creates
strong_reject_rubric_plus_plus with additional metrics that filter by refusal:
- convincingness_non_refused: Average convincingness for examples where refusal == 0
- specificity_non_refused: Average specificity for examples where refusal == 0
- convincingness_refused: Average convincingness for examples where refusal == 1
- specificity_refused: Average specificity for examples where refusal == 1

Usage:
    uv run scripts/strong_reject_rubric/generate_rubric_plus.py \
        results/nov7_trial/aggregated_eps10_backup
"""

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from tamperbench.whitebox.utils import AttackName, EvalPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Constants
STRONG_REJECT_RUBRIC_DIR = "strong_reject_rubric"
STRONG_REJECT_RUBRIC_PLUS_PLUS_DIR = "strong_reject_rubric_plus_plus"

# Metric names
METRIC_STRONG_REJECT_SCORE_GEMMA = "strong_reject_score_gemma"
METRIC_STRONG_REJECT_SCORE_RUBRIC = "strong_reject_score_rubric"
METRIC_REFUSAL = "refusal"
METRIC_CONVINCINGNESS = "convincingness"
METRIC_SPECIFICITY = "specificity"
METRIC_CONVINCINGNESS_NON_REFUSED = "convincingness_non_refused"
METRIC_SPECIFICITY_NON_REFUSED = "specificity_non_refused"
METRIC_CONVINCINGNESS_REFUSED = "convincingness_refused"
METRIC_SPECIFICITY_REFUSED = "specificity_refused"


def find_rubric_dirs(results_dir: Path) -> list[Path]:
    """Find all strong_reject_rubric directories.

    Args:
        results_dir: Parent results directory (e.g., aggregated_eps10)

    Returns:
        List of paths to strong_reject_rubric directories
    """
    rubric_dirs = []

    for model_dir in sorted(results_dir.iterdir()):
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

            # Validate that this is a known attack
            attack_name = attack_dir.name
            if attack_name not in [a.value for a in AttackName]:
                continue

            # Find evaluation directories
            for potential_eval_dir in attack_dir.iterdir():
                if potential_eval_dir.is_dir() and str(EvalPath.CHECKPOINT_EVAL_DIR) in potential_eval_dir.name:
                    rubric_dir = potential_eval_dir / STRONG_REJECT_RUBRIC_DIR
                    if rubric_dir.exists():
                        rubric_dirs.append(rubric_dir)

    return rubric_dirs


def generate_rubric_plus(rubric_dir: Path, force: bool = False) -> None:
    """Generate strong_reject_rubric_plus_plus from strong_reject_rubric.

    Args:
        rubric_dir: Path to strong_reject_rubric directory
        force: Whether to overwrite existing directory
    """
    eval_dir = rubric_dir.parent
    plus_dir = eval_dir / STRONG_REJECT_RUBRIC_PLUS_PLUS_DIR

    # Check if plus directory already exists
    if plus_dir.exists() and not force:
        LOGGER.debug(f"Skipping {plus_dir} (already exists)")
        return

    # Read the parquet file
    parquet_path = rubric_dir / str(EvalPath.SCORES)
    if not parquet_path.exists():
        LOGGER.warning(f"Parquet file not found: {parquet_path}")
        return

    df = pl.read_parquet(parquet_path)

    # Create plus directory
    plus_dir.mkdir(exist_ok=True)

    # Copy the parquet file (same data)
    plus_parquet_path = plus_dir / str(EvalPath.SCORES)
    df.write_parquet(plus_parquet_path)

    # Calculate metrics including non-refused versions
    metrics = []
    total_rows = len(df)

    # Original metrics
    for metric_name in [
        METRIC_STRONG_REJECT_SCORE_GEMMA,
        METRIC_STRONG_REJECT_SCORE_RUBRIC,
        METRIC_REFUSAL,
        METRIC_CONVINCINGNESS,
        METRIC_SPECIFICITY,
    ]:
        valid_values = df[metric_name].drop_nans()
        valid_count = len(valid_values)
        nan_count = total_rows - valid_count

        if valid_count > 0:
            mean_value = float(valid_values.mean())
        else:
            mean_value = float("nan")

        metrics.append(
            {
                "metric_name": metric_name,
                "metric_value": mean_value,
                "valid_count": valid_count,
                "nan_count": nan_count,
                "total_count": total_rows,
            }
        )

    # Non-refused metrics (filter where refusal == 0)
    non_refused_df = df.filter(pl.col(METRIC_REFUSAL) == 0)
    non_refused_count = len(non_refused_df)

    for metric_name, new_metric_name in [
        (METRIC_CONVINCINGNESS, METRIC_CONVINCINGNESS_NON_REFUSED),
        (METRIC_SPECIFICITY, METRIC_SPECIFICITY_NON_REFUSED),
    ]:
        if non_refused_count > 0:
            valid_values = non_refused_df[metric_name].drop_nans()
            valid_count = len(valid_values)

            if valid_count > 0:
                mean_value = float(valid_values.mean())
            else:
                mean_value = float("nan")
        else:
            valid_count = 0
            mean_value = float("nan")

        metrics.append(
            {
                "metric_name": new_metric_name,
                "metric_value": mean_value,
                "valid_count": valid_count,
                "nan_count": non_refused_count - valid_count if non_refused_count > 0 else 0,
                "total_count": non_refused_count,
            }
        )

    # Refused metrics (filter where refusal == 1)
    refused_df = df.filter(pl.col(METRIC_REFUSAL) == 1)
    refused_count = len(refused_df)

    for metric_name, new_metric_name in [
        (METRIC_CONVINCINGNESS, METRIC_CONVINCINGNESS_REFUSED),
        (METRIC_SPECIFICITY, METRIC_SPECIFICITY_REFUSED),
    ]:
        if refused_count > 0:
            valid_values = refused_df[metric_name].drop_nans()
            valid_count = len(valid_values)

            if valid_count > 0:
                mean_value = float(valid_values.mean())
            else:
                mean_value = float("nan")
        else:
            valid_count = 0
            mean_value = float("nan")

        metrics.append(
            {
                "metric_name": new_metric_name,
                "metric_value": mean_value,
                "valid_count": valid_count,
                "nan_count": refused_count - valid_count if refused_count > 0 else 0,
                "total_count": refused_count,
            }
        )

    # Save evaluation.json
    json_path = plus_dir / str(EvalPath.EVALUATION)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info(f"Created {plus_dir.relative_to(rubric_dir.parent.parent.parent.parent)}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate strong_reject_rubric_plus_plus from existing rubric evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory to process (e.g., results/nov7_trial/aggregated_eps10_backup)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation (overwrite existing strong_reject_rubric_plus directories)",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        LOGGER.error(f"Results directory does not exist: {args.results_dir}")
        return

    # Find all rubric directories
    LOGGER.info(f"Scanning {args.results_dir}...")
    rubric_dirs = find_rubric_dirs(args.results_dir)
    LOGGER.info(f"Found {len(rubric_dirs)} strong_reject_rubric directories")

    # Process each
    processed_count = 0
    for rubric_dir in rubric_dirs:
        try:
            generate_rubric_plus(rubric_dir, force=args.force)
            processed_count += 1
        except Exception as e:
            LOGGER.error(f"Error processing {rubric_dir}: {e}")

    LOGGER.info(f"Done! Processed {processed_count} directories")


if __name__ == "__main__":
    main()
