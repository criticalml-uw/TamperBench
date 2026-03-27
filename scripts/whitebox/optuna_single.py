"""Optuna runner limited to single-objective sweeps."""

# pyright: reportUnusedCallResult=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import multiprocessing
from argparse import Namespace
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import optuna
import torch

import tamperbench.whitebox.attacks  # ensure registry side-effects  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.evals  # noqa: F401 -- ensure registry side-effects  # pyright: ignore[reportUnusedImport]
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.utils import AttackName, ConfigPath
from tamperbench.whitebox.utils.benchmark.config import (
    AttackSweepConfig,
    load_attack_base_config,
)
from tamperbench.whitebox.utils.benchmark.path_generation import StudyPaths
from tamperbench.whitebox.utils.benchmark.runners import run_optuna_sweep
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager
from tamperbench.whitebox.utils.names import (
    ConfigKeys,
    EvalName,
)

REPO_ROOT = get_repo_root()

WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")


def _build_attack_objective(
    attack_name: AttackName,
    eval_names: list[EvalName],
    pretrained_model_path: str,
    study_paths: StudyPaths,
    random_seed: int,
) -> Callable[[optuna.trial.Trial, dict[str, object]], dict[str, float]]:
    """Build an objective function for attack sweeps.

    Creates a closure that runs a single attack trial via ``SweepTrialManager.run_trial``
    and returns evaluation metrics. This is passed as ``objective_fn`` to
    ``run_optuna_sweep``.

    Args:
        attack_name: Name of the attack to run in each trial.
        eval_names: Ordered list of evaluation names to compute metrics for.
        pretrained_model_path: Path to the pretrained model checkpoint.
        study_paths: Paths object for the current sweep study.
        random_seed: Random seed for reproducibility.

    Returns:
        A callable ``(trial, merged_config) -> dict[str, float]`` suitable for
        ``run_optuna_sweep``'s ``objective_fn`` parameter.
    """

    def attack_objective(trial: optuna.trial.Trial, merged_config: dict[str, object]) -> dict[str, float]:
        merged_config[ConfigKeys.EVALS] = [str(e) for e in eval_names]
        merged_config[ConfigKeys.RANDOM_SEED] = random_seed

        return SweepTrialManager.run_trial(
            attack_name=attack_name,
            attack_config_dict=merged_config,
            eval_names=eval_names,
            pretrained_model_path=pretrained_model_path,
            attack_results_dir=study_paths.sweep_results_dir,
            trial_number=trial.number,
            random_seed=random_seed,
        )

    return attack_objective


def main() -> None:
    """Run single-objective Optuna hyperparameter sweep over whitebox attacks."""
    parser = argparse.ArgumentParser(description="Optuna single-objective sweep over whitebox attacks")
    parser.add_argument(
        "pretrained_model_path",
        type=str,
        help="Path to an HF model or checkpoint",
    )
    parser.add_argument(
        "--attacks",
        type=AttackName,
        choices=list(AttackName),
        nargs="+",
        required=True,
        help="Attacks to sweep (space-separated)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=8,
        help="Number of Optuna trials per attack",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / f"optuna_single_{datetime.now():%Y_%m_%d}",
        help="Directory to store sweep results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top-performing configs to save in YAML (per attack)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=WHITEBOX_ATTACK_CONFIG_DIR,
        help=(
            "Optional directory containing whitebox attack configs (each attack subfolder with grid.yaml & sweep files). "
            "Defaults to repo configs/whitebox/attacks"
        ),
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
        help="Random seed for reproducibility (used by attacks and Optuna sampler)",
    )

    args: Namespace = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)

    config_root = cast(Path, args.configs_dir)
    pretrained_model_path = cast(str, args.pretrained_model_path)
    results_dir = cast(Path, args.results_dir)
    model_alias = cast(
        str,
        args.model_alias or f"{Path(pretrained_model_path).name}_{datetime.now():%Y_%m_%d}",
    )
    random_seed = cast(int, args.random_seed)

    for attack_name in cast(list[AttackName], args.attacks):
        base_name, base_attack_config = load_attack_base_config(config_root, attack_name)
        sweep_config = AttackSweepConfig.from_yaml(
            config_root, attack_name, sweep_filename=ConfigPath.SINGLE_OBJECTIVE_SWEEP
        )

        study_paths = StudyPaths(
            results_dir=results_dir,
            model_alias=model_alias,
            attack_name=attack_name,
        )

        attack_objective = _build_attack_objective(
            attack_name=attack_name,
            eval_names=sweep_config.evals,
            pretrained_model_path=pretrained_model_path,
            study_paths=study_paths,
            random_seed=random_seed,
        )

        run_optuna_sweep(
            study_paths=study_paths,
            sweep_space=sweep_config.sweep,
            base_config=base_attack_config,
            direction=sweep_config.primary_objective_direction,
            objective_fn=attack_objective,
            primary_metric_key=str(sweep_config.evals[0]),
            n_trials=cast(int, args.n_trials),
            random_seed=random_seed,
            top_n=cast(int, args.top_n),
            eval_names=sweep_config.evals,
            base_config_name=base_name,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
