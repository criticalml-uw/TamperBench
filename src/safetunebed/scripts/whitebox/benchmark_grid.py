"""Script to benchmark an LLM against a suite of tampering attacks with configs defined in a grid."""

# pyright: reportUnusedCallResult=false, reportAny=false

import argparse
import multiprocessing
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from safetunebed.utils import get_repo_root
from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.attacks.embedding_attack.embedding_attack import (
    EmbeddingAttack,
    EmbeddingAttackConfig,
)
from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from safetunebed.whitebox.attacks.multilingual_finetune.multilingual_finetune import (
    MultilingualFinetune,
)
from safetunebed.whitebox.utils import AttackName, ConfigPath
from safetunebed.whitebox.utils.benchmark.io import yaml_to_dict

REPO_ROOT = get_repo_root()

WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")

ATTACKS_MAP = {
    AttackName.EMBEDDING_ATTACK: (EmbeddingAttackConfig, EmbeddingAttack),
    AttackName.FULL_PARAMETER_FINETUNE: (
        FullParameterFinetuneConfig,
        FullParameterFinetune,
    ),
    AttackName.LORA_FINETUNE: (LoraFinetuneConfig, LoraFinetune),
    AttackName.MULTILINGUAL_FINETUNE: (
        FullParameterFinetuneConfig,
        MultilingualFinetune,
    ),
}


def _benchmark_attacker(
    attack_name: AttackName,
    config_name: str,
    attack_config_dict: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    pretrained_model_path: str,
    results_dir: str,
) -> None:
    """Benchmark an attack method for a given config."""
    attack_config_cls, attack_cls = ATTACKS_MAP[attack_name]
    attacker: TamperAttack[TamperAttackConfig] = attack_cls(  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
        attack_config=attack_config_cls.from_dict(  # pyright: ignore[reportArgumentType]
            data={
                "input_checkpoint_path": pretrained_model_path,
                "out_dir": Path(results_dir, attack_name, config_name),
                **attack_config_dict,
            }
        )
    )
    attacker.benchmark()


whitebox_attack_config_grids: dict[AttackName, dict[str, Any]] = {  # pyright: ignore[reportExplicitAny]
    attack_name: yaml_to_dict(
        Path(WHITEBOX_ATTACK_CONFIG_DIR, attack_name, ConfigPath.GRID_YAML)
    )
    for attack_name in ATTACKS_MAP.keys()
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM against a suite of tampering attacks."
    )

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
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / f"grid_{datetime.now():%Y_%m_%d}",  # computed below if not provided
        help="Directory to store results. Default: results/grid_yyyy_mm_dd.",
    )

    args: Namespace = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)

    for attack_name in args.attacks:
        attack_config_cls, attack_cls = ATTACKS_MAP[attack_name]
        for config_name, attack_config_dict in whitebox_attack_config_grids[
            attack_name
        ].items():
            p = multiprocessing.Process(
                target=_benchmark_attacker,
                args=(
                    attack_name,
                    config_name,
                    attack_config_dict,
                    args.pretrained_model_path,
                    args.results_dir,
                ),
            )
            p.start()
            p.join()

            torch.cuda.empty_cache()
