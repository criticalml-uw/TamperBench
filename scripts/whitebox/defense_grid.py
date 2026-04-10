"""Script to benchmark a defense against attacks with configs defined in a grid."""

# pyright: reportUnusedCallResult=false, reportAny=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from dotenv import load_dotenv

# Import to trigger registry side-effects
import tamperbench.whitebox.attacks  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.defenses  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.evals  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.utils import ConfigPath
from tamperbench.whitebox.utils.benchmark.defense_config import (
    DefenseGridConfig,
)
from tamperbench.whitebox.utils.benchmark.defense_trial_manager import DefenseSweepTrialManager
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.names import DefenseName

REPO_ROOT = get_repo_root()
WHITEBOX_DEFENSE_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "defenses")
WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")


def main() -> None:
    """Run grid-based defense benchmark."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark a defense against attacks with grid configs.")
    parser.add_argument(
        "pretrained_model_path",
        type=str,
        help="Path to an HF model or checkpoint",
    )
    parser.add_argument(
        "--defense",
        type=DefenseName,
        choices=list(DefenseName),
        required=True,
        help="Defense to benchmark",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name from grid.yaml (default: all configs)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / f"defense_grid_{datetime.now():%Y_%m_%d}",
        help="Directory to store results",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=WHITEBOX_DEFENSE_CONFIG_DIR,
        help="Directory containing defense configs",
    )
    parser.add_argument(
        "--attack-configs-dir",
        type=Path,
        default=WHITEBOX_ATTACK_CONFIG_DIR,
        help="Directory containing attack configs",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=None,
        help="Alias for the model (default: {model_basename}_{date})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        default=False,
        help="Keep defended model checkpoints instead of deleting them after evaluation",
    )

    args: Namespace = parser.parse_args()

    config_root = cast(Path, args.configs_dir)
    attack_configs_dir = cast(Path, args.attack_configs_dir)
    pretrained_model_path = cast(str, args.pretrained_model_path)
    results_dir = cast(Path, args.results_dir)
    defense_name = cast(DefenseName, args.defense)
    model_alias = cast(
        str,
        args.model_alias or f"{Path(pretrained_model_path).name}_{datetime.now():%Y_%m_%d}",
    )
    random_seed = cast(int, args.random_seed)

    # Load the full grid
    grid_path = Path(config_root, defense_name, ConfigPath.GRID_YAML)
    defense_grid: dict[str, object] = yaml_to_dict(grid_path)

    # Filter to specific config if requested
    if args.config_name:
        config_name_str = cast(str, args.config_name)
        if config_name_str not in defense_grid:
            raise ValueError(f"Config '{config_name_str}' not found. Available: {list(defense_grid.keys())}")
        defense_grid = {config_name_str: defense_grid[config_name_str]}

    config_names = list(defense_grid.keys())

    for config_name, config_value in defense_grid.items():
        if not isinstance(config_value, dict):
            raise TypeError(f"Expected dict for config '{config_name}', got {type(config_value)}")

        config_dict: dict[str, object] = config_value

        # Parse meta-fields
        grid_config = DefenseGridConfig.from_dict(config_dict)

        print(f"\n{'=' * 60}")
        print(f"Running defense: {defense_name}, config: {config_name}")
        print(f"{'=' * 60}\n")

        # Call DefenseSweepTrialManager.run_trial for each config
        metrics = DefenseSweepTrialManager.run_trial(
            defense_name=defense_name,
            defense_config_dict=config_dict,
            defense_eval_names=grid_config.defense_evals,
            post_attack_eval_names=grid_config.post_attack_evals,
            model_config_dict=grid_config.inference_model_config,
            attacks=grid_config.attacks,
            pretrained_model_path=pretrained_model_path,
            defense_results_dir=results_dir / model_alias / str(defense_name) / "grid",
            trial_number=config_names.index(config_name),
            random_seed=random_seed,
            attack_configs_dir=attack_configs_dir,
            model_alias=model_alias,
            cleanup_checkpoints=not args.keep_checkpoints,
        )

        print(f"\nMetrics for {config_name}:")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
