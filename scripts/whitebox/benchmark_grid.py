"""Script to benchmark an LLM against a suite of tampering attacks with configs defined in a grid."""

# pyright: reportUnusedCallResult=false, reportAny=false, reportDuplicateImport=false, reportUnknownVariableType=false

import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv

from tamperbench.utils import get_repo_root
from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.embedding_attack import (
    embedding_attack as _,
)
from tamperbench.whitebox.attacks.full_parameter_finetune import (
    full_parameter_finetune as _,
)
from tamperbench.whitebox.attacks.jailbreak_finetune import (
    jailbreak_finetune as _,
)
from tamperbench.whitebox.attacks.lora_finetune import lora_finetune as _
from tamperbench.whitebox.attacks.multilingual_finetune import (
    multilingual_finetune as _,  # noqa: F401
)
from tamperbench.whitebox.attacks.registry import (
    ATTACKS_REGISTRY,
)
from tamperbench.whitebox.utils import AttackName, ConfigPath
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.names import ConfigKeys

REPO_ROOT = get_repo_root()

WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark an LLM against a suite of tampering attacks.")

    parser.add_argument(
        "pretrained_model_path",
        type=str,
        help="String path to a Hugging Face pre-trained model or HF Transformers checkpoint.",
    )
    parser.add_argument(
        "--attacks",
        type=AttackName,
        choices=list(AttackName),
        nargs="+",
        help=(
            "List of attacks to run (case-sensitive). Add spaces between attack names, e.g.: "
            "'--attacks lora_finetune embedding_attack'"
        ),
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / f"grid_{datetime.now():%Y_%m_%d}",
        help="Directory to store results. Default: results/grid_yyyy_mm_dd.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing whitebox attack configs (each attack subfolder with grid.yaml). "
            "Defaults to repo configs/whitebox/attacks"
        ),
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=None,
        help="Alias for the model (default: {model_basename}_{date}). Results stored under results_dir/model_alias/",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args: Namespace = parser.parse_args()

    config_root = args.configs_dir if args.configs_dir else WHITEBOX_ATTACK_CONFIG_DIR
    pretrained_model_path = cast(str, args.pretrained_model_path)
    results_dir = cast(Path, args.results_dir)
    model_alias = cast(
        str,
        args.model_alias or f"{Path(pretrained_model_path).name}_{datetime.now():%Y_%m_%d}",
    )
    random_seed = cast(int, args.random_seed)

    whitebox_attack_config_grids: dict[AttackName, dict[str, Any]] | None = {
        attack_name: yaml_to_dict(Path(config_root, attack_name, ConfigPath.GRID_YAML)) for attack_name in args.attacks
    }

    for attack_name in args.attacks:
        attack_config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]
        for config_name, attack_config_dict in whitebox_attack_config_grids[attack_name].items():
            # Inject random seed into config
            attack_config_dict[ConfigKeys.RANDOM_SEED] = random_seed

            attacker: TamperAttack[TamperAttackConfig] = attack_cls(
                attack_config=attack_config_cls.from_dict(
                    data={
                        "input_checkpoint_path": pretrained_model_path,
                        "out_dir": Path(results_dir, model_alias, attack_name, config_name),
                        **attack_config_dict,
                    }
                )
            )
            attacker.benchmark()

        torch.cuda.empty_cache()
