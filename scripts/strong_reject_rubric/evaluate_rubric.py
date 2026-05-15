#!/usr/bin/env python3
"""Evaluate strong-reject results using the rubric-based evaluator.

This script processes epsilon-bounded results directories and adds rubric-based
strong-reject evaluations alongside existing Gemma-based evaluations.

The script includes automatic retry logic for NaN scores, which can help handle
transient API failures or rate limiting issues.

Usage:
    # Process a results directory with default retry (2 attempts)
    uv run scripts/strong_reject_rubric/evaluate_rubric.py \
        results/nov7_trial/aggregated_eps10

    # Force re-evaluation (overwrite existing)
    uv run scripts/strong_reject_rubric/evaluate_rubric.py \
        results/nov7_trial/aggregated_eps10 \
        --force

    # Configure retry attempts for NaN scores
    uv run scripts/strong_reject_rubric/evaluate_rubric.py \
        results/nov7_trial/aggregated_eps10 \
        --max-retries 3

    # Customize both workers and retries
    uv run scripts/strong_reject_rubric/evaluate_rubric.py \
        results/nov7_trial/aggregated_eps10 \
        --max-workers 20 \
        --max-retries 3
"""

import argparse
import logging
from pathlib import Path

from rubric_utils import (
    find_evaluation_dirs,
    should_process,
    evaluate_with_rubric,
    save_rubric_results,
)
from tamperbench.whitebox.utils import EvalPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate strong-reject results using rubric-based evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory to process (e.g., results/nov7_trial/aggregated_eps10)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation (overwrite existing strong_reject_rubric directories)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retry attempts for NaN scores (default: 2)",
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        LOGGER.error(f"Results directory does not exist: {args.results_dir}")
        return

    if not args.results_dir.is_dir():
        LOGGER.error(f"Results path is not a directory: {args.results_dir}")
        return

    # Find all evaluation directories
    LOGGER.info(f"Scanning {args.results_dir}...")
    eval_dirs = find_evaluation_dirs(args.results_dir)

    # Filter directories that should be processed
    to_process = []
    skipped = []
    for eval_dir in eval_dirs:
        should_proc, reason = should_process(eval_dir, args.force)
        if should_proc:
            to_process.append(eval_dir)
        else:
            skipped.append((eval_dir, reason))

    LOGGER.info(f"Found {len(eval_dirs)} evaluations: {len(to_process)} to process, {len(skipped)} to skip")

    # Process each evaluation directory
    processed_count = 0
    error_count = 0

    for i, eval_dir in enumerate(to_process, 1):
        rel_path = eval_dir.relative_to(args.results_dir)
        LOGGER.info(f"[{i}/{len(to_process)}] Processing {rel_path}")

        try:
            # Read original parquet
            parquet_path = eval_dir / "strong_reject" / str(EvalPath.SCORES)

            # Evaluate with rubric
            rubric_df = evaluate_with_rubric(parquet_path, max_workers=args.max_workers, max_retries=args.max_retries)

            # Save results
            save_rubric_results(eval_dir, rubric_df, force=args.force)

            processed_count += 1

        except Exception as e:
            error_count += 1
            LOGGER.error(f"Error processing {rel_path}: {e}")
            continue

    # Print summary
    LOGGER.info(f"Complete: {processed_count} processed, {len(skipped)} skipped, {error_count} errors")


if __name__ == "__main__":
    main()
