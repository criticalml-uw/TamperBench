#!/usr/bin/env python3
"""Main analysis script for epsilon-bounded analysis of Optuna sweep results.

Usage:
    # Epsilon-bounded analysis
    uv run scripts/analysis/analyze_results.py results/sweep epsilon \
        --epsilons 0.05 0.20 0.50

    # With custom config
    uv run scripts/analysis/analyze_results.py results/sweep epsilon \
        --config configs/analysis/custom.yaml \
        --epsilons 0.05
"""

import argparse
import logging
from pathlib import Path

from tamperbench.whitebox.utils.analysis import AnalysisConfig
from tamperbench.whitebox.utils.analysis.pipeline import AnalysisPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run epsilon-bounded analysis on Optuna sweep results."""
    parser = argparse.ArgumentParser(
        description="Analyze Optuna sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Parent results directory containing model subdirectories",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/default.yaml"),
        help="Analysis configuration file (default: configs/analysis/default.yaml)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Epsilon-bounded command
    eps = subparsers.add_parser(
        "epsilon",
        help="Run epsilon-bounded filtering and analysis",
    )
    eps.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.05, 0.20, 0.50],
        help="Epsilon bounds (MMLU drop thresholds) (default: 0.05 0.20 0.50)",
    )
    eps.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of chronological trials to consider (default: 40)",
    )
    eps.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip aggregation step (only filter trials)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {args.results_dir}")

    if not args.config.exists():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")

    LOGGER.info(f"Loading configuration from {args.config}")
    config = AnalysisConfig.from_yaml(args.config)

    pipeline = AnalysisPipeline(config)

    try:
        LOGGER.info(f"Running epsilon-bounded analysis with epsilons: {args.epsilons}")
        pipeline.run_epsilon_bounded(
            args.results_dir,
            args.epsilons,
            args.n_trials,
            args.skip_aggregation,
        )
        LOGGER.info("Analysis complete!")

    except Exception as e:
        LOGGER.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
