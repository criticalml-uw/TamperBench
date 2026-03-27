"""Script to run alignment defenses with configs defined in YAML files."""

import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import git

from tamperbench.whitebox.defenses.defense import AlignmentDefense, AlignmentDefenseConfig
from tamperbench.whitebox.defenses.registry import DEFENSES_REGISTRY

# Import to trigger defense registration
from tamperbench.whitebox.defenses.t_vaccine import tar as _  # noqa: F401
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.names import ConfigPath, DefenseName

REPO_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir  # pyright: ignore[reportArgumentType]
)

WHITEBOX_DEFENSE_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "defenses")


def main() -> None:
    """Run a defense with a config from YAML."""
    parser = argparse.ArgumentParser(description="Run an alignment defense on a model.")

    parser.add_argument(
        "pretrained_model_path",
        type=str,
        help="String path to a Hugging Face pre-trained model or HF Transformers checkpoint.",
    )
    parser.add_argument(
        "--defense",
        type=DefenseName,
        choices=list(DefenseName),
        required=True,
        help="Defense to run (e.g., 'tar')",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="base",
        help="Config name from grid.yaml (default: 'base')",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to store aligned checkpoint. Default: results/defense_<name>_<date>",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing whitebox defense configs (each defense subfolder with grid.yaml). "
            "Defaults to repo configs/whitebox/defenses"
        ),
    )

    args: Namespace = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "results" / f"defense_{args.defense}_{datetime.now():%Y_%m_%d_%H%M%S}"

    # Load defense config from YAML
    config_root = args.configs_dir if args.configs_dir else WHITEBOX_DEFENSE_CONFIG_DIR
    defense_config_path = Path(config_root, args.defense, ConfigPath.GRID_YAML)

    if not defense_config_path.exists():
        raise FileNotFoundError(
            f"Defense config not found at {defense_config_path}. "
            f"Make sure configs/whitebox/defenses/{args.defense}/grid.yaml exists."
        )

    defense_config_grid = yaml_to_dict(defense_config_path)

    # Get the specific config (default to "base")
    if args.config_name not in defense_config_grid:
        raise ValueError(
            f"Config '{args.config_name}' not found in {defense_config_path}. "
            f"Available configs: {list(defense_config_grid.keys())}"
        )

    defense_config_dict: dict[str, object] = defense_config_grid[args.config_name]  # pyright: ignore[reportAssignmentType]

    # Get defense class and config class from registry
    if args.defense not in DEFENSES_REGISTRY:
        raise ValueError(
            f"Defense '{args.defense}' not registered. Available defenses: {list(DEFENSES_REGISTRY.keys())}"
        )

    defense_config_cls, defense_cls = DEFENSES_REGISTRY[args.defense]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Construct defense config
    print(f"\n{'=' * 60}")
    print(f"Running defense: {args.defense}")
    print(f"Config: {args.config_name}")
    print(f"Input model: {args.pretrained_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 60}\n")

    # Filter out fields that are not part of defense config (model_config, evals are for attacks/evaluation)
    filtered_config_dict = {
        k: v
        for k, v in defense_config_dict.items()
        if k not in ["model_config", "evals", "defense_evals", "post_attack_evals", "attacks"]
    }

    defense_config: AlignmentDefenseConfig = defense_config_cls.from_dict(
        data={
            "input_checkpoint_path": args.pretrained_model_path,
            "output_checkpoint_path": args.output_dir,
            **filtered_config_dict,
        }
    )

    # Run defense
    defense: AlignmentDefense[AlignmentDefenseConfig] = defense_cls(defense_config=defense_config)
    aligned_checkpoint_path = defense.run_defense()

    print(f"\n{'=' * 60}")
    print("✓ Defense completed successfully!")
    print(f"Aligned checkpoint saved to: {aligned_checkpoint_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
