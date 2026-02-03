"""Optuna runner limited to single-objective sweeps."""

# pyright: reportUnusedCallResult=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import copy
import multiprocessing
import traceback
from argparse import Namespace
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
from tamperbench.whitebox.utils.benchmark.io import deep_merge_dicts
from tamperbench.whitebox.utils.benchmark.path_generation import StudyPaths
from tamperbench.whitebox.utils.benchmark.sweep import suggest_nested_params
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager
from tamperbench.whitebox.utils.names import (
    ConfigKeys,
    EvalName,
    OptunaUserAttrs,
    TrialFailureKeys,
)

REPO_ROOT = get_repo_root()

WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")


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
        study_paths.ensure_dirs()

        # Use TPESampler with seed for reproducibility
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(
            direction=sweep_config.primary_objective_direction,
            study_name=study_paths.study_name,
            storage=study_paths.storage_url,
            load_if_exists=True,
            sampler=sampler,
        )

        def objective(
            trial: optuna.trial.Trial,
            *,
            _sweep_space: dict[str, object] = sweep_config.sweep,
            _base_attack_config: dict[str, object] = base_attack_config,
            _eval_names: list[EvalName] = sweep_config.evals,
            _attack_name: AttackName = attack_name,
            _pretrained_model_path: str = pretrained_model_path,
            _study_paths: StudyPaths = study_paths,
            _random_seed: int = random_seed,
        ) -> float:
            trial_overrides = suggest_nested_params(trial, _sweep_space)
            merged_config = deep_merge_dicts(_base_attack_config, trial_overrides)
            merged_config[ConfigKeys.EVALS] = [str(e) for e in _eval_names]
            merged_config[ConfigKeys.RANDOM_SEED] = _random_seed

            trial.set_user_attr(OptunaUserAttrs.CONFIG_OVERRIDES, trial_overrides)
            trial.set_user_attr(OptunaUserAttrs.MERGED_CONFIG, copy.deepcopy(merged_config))

            try:
                objective_values = SweepTrialManager.run_trial(
                    attack_name=_attack_name,
                    attack_config_dict=merged_config,
                    eval_names=_eval_names,
                    pretrained_model_path=_pretrained_model_path,
                    attack_results_dir=_study_paths.attack_results_dir,
                    trial_number=trial.number,
                    random_seed=_random_seed,
                )
                trial.set_user_attr(OptunaUserAttrs.EVAL_METRICS, objective_values)
                return objective_values[_eval_names[0]]

            except Exception as error:  # pragma: no cover - surfaced to study logs
                stack_trace = traceback.format_exc()
                trial.set_user_attr(
                    OptunaUserAttrs.FAILURE,
                    {
                        TrialFailureKeys.ERROR: str(error),
                        TrialFailureKeys.TRACEBACK: stack_trace,
                    },
                )
                raise optuna.exceptions.TrialPruned(f"Trial failed due to exception: {error}") from error

        study.optimize(objective, n_trials=cast(int, args.n_trials))

        SweepTrialManager.summarize_top_n(
            study,
            direction=sweep_config.primary_objective_direction,
            top_n=cast(int, args.top_n),
            eval_names=sweep_config.evals,
            base_cfg=base_attack_config,
            base_config_name=base_name,
            output_dir=study_paths.attack_results_dir,
            study_storage=study_paths.storage_path,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
