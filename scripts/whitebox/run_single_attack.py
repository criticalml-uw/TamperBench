"""Run a single attack with a single config against a pretrained model."""

# pyright: reportUnusedCallResult=false, reportAny=false, reportDuplicateImport=false, reportUnknownVariableType=false

import argparse
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from dotenv import load_dotenv

from tamperbench.utils import get_repo_root
from tamperbench.whitebox.attacks.embedding_attack import embedding_attack as _
from tamperbench.whitebox.attacks.full_parameter_finetune import full_parameter_finetune as _
from tamperbench.whitebox.attacks.jailbreak_finetune import jailbreak_finetune as _
from tamperbench.whitebox.attacks.lora_finetune import lora_finetune as _
from tamperbench.whitebox.attacks.multilingual_finetune import multilingual_finetune as _  # noqa: F401
from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY
from tamperbench.whitebox.utils import AttackName, ConfigPath
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.benchmark.runners import run_attack_grid_configs

REPO_ROOT = get_repo_root()
WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a single attack with a single config.")
    parser.add_argument("pretrained_model_path", type=str, help="HF model path or checkpoint.")
    parser.add_argument("--attack", type=AttackName, choices=list(AttackName), required=True)
    parser.add_argument("--config-name", type=str, default="base", help="Config entry in grid YAML (default: 'base').")
    parser.add_argument("--configs-dir", type=Path, default=None, help="Override config directory.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--model-alias", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cleanup-checkpoints", action="store_true")
    args = parser.parse_args()

    pretrained_model_path = cast(str, args.pretrained_model_path)
    attack_name: AttackName = args.attack
    results_dir = cast(Path, args.results_dir or REPO_ROOT / "results" / f"single_{datetime.now():%Y_%m_%d}")
    model_alias = cast(str, args.model_alias or f"{Path(pretrained_model_path).name}_{datetime.now():%Y_%m_%d}")

    config_root = args.configs_dir if args.configs_dir else WHITEBOX_ATTACK_CONFIG_DIR
    grid = yaml_to_dict(Path(config_root, attack_name, ConfigPath.GRID_YAML))

    config_name: str = args.config_name
    if config_name not in grid:
        raise KeyError(f"Config '{config_name}' not found. Available: {', '.join(grid.keys())}")

    single_config_grid = {config_name: grid[config_name]}

    results = run_attack_grid_configs(
        attack_name=attack_name,
        config_grid=single_config_grid,
        pretrained_model_path=pretrained_model_path,
        output_base_dir=Path(results_dir, model_alias, attack_name),
        random_seed=cast(int, args.random_seed),
        cleanup_checkpoints=args.cleanup_checkpoints,
    )

    print(results[config_name])
    torch.cuda.empty_cache()
