"""Optuna runner for single-objective defense hyperparameter sweeps."""

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

# Import to trigger registry side-effects
import tamperbench.whitebox.attacks  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.defenses  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.evals  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.utils import ConfigPath
from tamperbench.whitebox.utils.benchmark.defense_config import (
    CapabilityGuard,
    DefenseSweepConfig,
    PostDefenseAttackSpec,
    load_defense_base_config,
)
from tamperbench.whitebox.utils.benchmark.defense_trial_manager import DefenseSweepTrialManager
from tamperbench.whitebox.utils.benchmark.path_generation import DefenseStudyPaths
from tamperbench.whitebox.utils.benchmark.runners import run_optuna_sweep
from tamperbench.whitebox.utils.names import DefenseMetricPrefix, DefenseName, EvalName

REPO_ROOT = get_repo_root()
WHITEBOX_DEFENSE_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "defenses")
WHITEBOX_ATTACK_CONFIG_DIR = REPO_ROOT / Path("configs", "whitebox", "attacks")


def _build_defense_objective(
    defense_name: DefenseName,
    defense_eval_names: list[EvalName],
    post_attack_eval_names: list[EvalName],
    model_config_dict: dict[str, object],
    attacks: list[PostDefenseAttackSpec],
    pretrained_model_path: str,
    study_paths: DefenseStudyPaths,
    random_seed: int,
    attack_configs_dir: Path | None = None,
    model_alias: str = "",
    capability_guards: dict[EvalName, CapabilityGuard] | None = None,
    capability_baselines: dict[str, float] | None = None,
    cleanup_checkpoints: bool = True,
) -> Callable[[optuna.trial.Trial, dict[str, object]], dict[str, float]]:
    """Build objective function for defense sweep.

    Creates a closure that runs a single defense trial via
    ``DefenseSweepTrialManager.run_trial`` and returns evaluation metrics.
    This is passed as ``objective_fn`` to ``run_optuna_sweep``.

    The returned dict has prefixed keys like ``"defense.strong_reject"``,
    ``"post_attack.lora_finetune.strong_reject"``.

    Args:
        defense_name: Name of the defense to run in each trial.
        defense_eval_names: Evaluations to run on the defense checkpoint.
        post_attack_eval_names: Evaluations to run on post-attack checkpoints.
        model_config_dict: Raw model configuration dict for inference.
        attacks: Attack specifications to stress-test the defense.
        pretrained_model_path: Path to the pretrained model checkpoint.
        study_paths: Paths object for the current defense sweep study.
        random_seed: Random seed for reproducibility.
        attack_configs_dir: Root directory for attack configs used in
            post-defense attacks.
        model_alias: Alias for the model, passed to inner attack sweeps.
        capability_guards: Optional retention thresholds from sweep config.
        capability_baselines: Base model scores for guarded metrics.
        cleanup_checkpoints: Whether to delete model checkpoints after evaluation.

    Returns:
        A callable ``(trial, merged_config) -> dict[str, float]`` suitable for
        ``run_optuna_sweep``'s ``objective_fn`` parameter.
    """

    def defense_objective(trial: optuna.trial.Trial, merged_config: dict[str, object]) -> dict[str, float]:
        return DefenseSweepTrialManager.run_trial(
            defense_name=defense_name,
            defense_config_dict=merged_config,
            defense_eval_names=defense_eval_names,
            post_attack_eval_names=post_attack_eval_names,
            model_config_dict=model_config_dict,
            attacks=attacks,
            pretrained_model_path=pretrained_model_path,
            defense_results_dir=study_paths.sweep_results_dir,
            trial_number=trial.number,
            random_seed=random_seed,
            attack_configs_dir=attack_configs_dir,
            model_alias=model_alias,
            capability_guards=capability_guards,
            capability_baselines=capability_baselines,
            cleanup_checkpoints=cleanup_checkpoints,
        )

    return defense_objective


def main() -> None:
    """Run single-objective Optuna hyperparameter sweep over a defense."""
    parser = argparse.ArgumentParser(description="Optuna single-objective sweep over a defense")
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
        help="Defense to sweep",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=8,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / f"defense_sweep_{datetime.now():%Y_%m_%d}",
        help="Directory to store sweep results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top configs to save in YAML",
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
        help="Directory containing attack configs (for post-defense attacks)",
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
    multiprocessing.set_start_method("spawn", force=True)

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

    # Load base config and sweep config
    base_name, base_defense_config = load_defense_base_config(config_root, defense_name)
    sweep_config = DefenseSweepConfig.from_yaml(
        config_root, defense_name, sweep_filename=ConfigPath.SINGLE_OBJECTIVE_SWEEP
    )

    study_paths = DefenseStudyPaths(
        results_dir=results_dir,
        model_alias=model_alias,
        defense_name=defense_name,
    )

    # Build prefixed metric keys so best.json values are populated correctly
    attack_names = [attack.name for attack in sweep_config.attacks]
    all_metric_keys = DefenseMetricPrefix.build_all_metric_keys(
        defense_eval_names=sweep_config.defense_evals,
        post_attack_eval_names=sweep_config.post_attack_evals,
        attack_names=attack_names,
    )

    # Primary metric: first post_attack_eval from first attack (prefixed key)
    first_attack = sweep_config.attacks[0]
    primary_metric_key = DefenseMetricPrefix.post_attack_key(first_attack.name, sweep_config.post_attack_evals[0])

    # Evaluate base model on guarded metrics (if any guards are configured)
    capability_baselines: dict[str, float] | None = None
    if sweep_config.capability_guards:
        from tamperbench.whitebox.utils.models.config import ModelConfig

        baseline_dir = study_paths.sweep_results_dir / "capability_baselines"
        capability_baselines = DefenseSweepTrialManager.evaluate_baselines(
            pretrained_model_path=pretrained_model_path,
            capability_guards=sweep_config.capability_guards,
            model_config=ModelConfig.from_dict(dict(sweep_config.inference_model_config)),
            out_dir=baseline_dir,
        )

    defense_objective = _build_defense_objective(
        defense_name=defense_name,
        defense_eval_names=sweep_config.defense_evals,
        post_attack_eval_names=sweep_config.post_attack_evals,
        model_config_dict=sweep_config.inference_model_config,
        attacks=sweep_config.attacks,
        pretrained_model_path=pretrained_model_path,
        study_paths=study_paths,
        random_seed=random_seed,
        attack_configs_dir=attack_configs_dir,
        model_alias=model_alias,
        capability_guards=sweep_config.capability_guards if sweep_config.capability_guards else None,
        capability_baselines=capability_baselines,
        cleanup_checkpoints=not args.keep_checkpoints,
    )

    run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_config.sweep,
        base_config=base_defense_config,
        direction=sweep_config.primary_objective_direction,
        objective_fn=defense_objective,
        primary_metric_key=primary_metric_key,
        n_trials=cast(int, args.n_trials),
        random_seed=random_seed,
        top_n=cast(int, args.top_n),
        eval_names=all_metric_keys,
        base_config_name=base_name,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
