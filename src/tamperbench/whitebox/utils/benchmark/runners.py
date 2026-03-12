"""Reusable runner functions for Optuna sweeps and grid-based attack benchmarks.

Extracts shared logic from ``scripts/whitebox/optuna_single.py`` (Optuna loop) and
``scripts/whitebox/benchmark_grid.py`` (grid loop) so both attack scripts and future
defense scripts can share the same core without duplicating boilerplate.
"""

# pyright: reportUnusedCallResult=false, reportUnknownVariableType=false

from __future__ import annotations

import copy
import logging
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import optuna

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY
from tamperbench.whitebox.utils.benchmark.io import deep_merge_dicts
from tamperbench.whitebox.utils.benchmark.sweep import suggest_nested_params
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager
from tamperbench.whitebox.utils.names import (
    AttackName,
    ConfigKeys,
    EvalName,
    OptunaUserAttrs,
    TrialFailureKeys,
)

if TYPE_CHECKING:
    from pandera.typing.polars import DataFrame

    from tamperbench.whitebox.evals.output_schema import EvaluationSchema

logger = logging.getLogger(__name__)


@runtime_checkable
class StudyPathsLike(Protocol):
    """Protocol for objects providing paths required by ``run_optuna_sweep``.

    Both ``StudyPaths`` (attack sweeps) and ``DefenseStudyPaths`` satisfy
    this protocol, allowing ``run_optuna_sweep`` to work with either.
    """

    @property
    def sweep_results_dir(self) -> Path:
        """Directory where trial results, study DB, and summaries are stored."""
        ...

    @property
    def storage_path(self) -> Path:
        """Path to the Optuna study SQLite database file."""
        ...

    @property
    def storage_url(self) -> str:
        """SQLite connection URL for Optuna study storage."""
        ...

    @property
    def study_name(self) -> str:
        """Optuna study name."""
        ...

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        ...


def run_optuna_sweep(
    study_paths: StudyPathsLike,
    sweep_space: dict[str, object],
    base_config: dict[str, object],
    direction: str,
    objective_fn: Callable[[optuna.trial.Trial, dict[str, object]], dict[str, float]],
    primary_metric_key: str,
    n_trials: int,
    random_seed: int,
    top_n: int,
    eval_names: Sequence[str],
    base_config_name: str,
) -> optuna.study.Study:
    """Run a single-objective Optuna sweep and summarize results.

    Shared core for attack sweeps, defense sweeps, AND inner attack sweeps
    within defense trials. The caller provides ``objective_fn`` which receives
    ``(trial, merged_config)`` and returns ``{metric_key: value}``.

    Steps:
      1. Create Optuna study (SQLite-backed, TPESampler, ``load_if_exists``).
      2. For each trial:
         a. ``suggest_nested_params(trial, sweep_space)``
         b. ``deep_merge_dicts(base_config, overrides)``
         c. Store config overrides + merged config in ``trial.user_attrs``
         d. Call ``objective_fn(trial, merged_config)`` -> ``{metric_key: float}``
         e. Store metrics in ``trial.user_attrs``
         f. Return ``metrics[primary_metric_key]`` to Optuna
         g. On error: store traceback, raise ``TrialPruned``
      3. ``study.optimize(objective, callbacks=[MaxTrialsCallback(n_trials)])``
      4. ``SweepTrialManager.summarize_top_n()``
      5. Return study

    Args:
        study_paths: Object providing paths for the study directory, storage URL,
            and study name. Must satisfy the ``StudyPathsLike`` protocol.
        sweep_space: Nested Optuna search space definition (from sweep YAML).
        base_config: Base configuration dictionary to merge with suggested params.
        direction: Optimization direction, either ``"maximize"`` or ``"minimize"``.
        objective_fn: Callback receiving ``(trial, merged_config_dict)`` and
            returning ``dict[str, float]`` mapping metric keys to values.
        primary_metric_key: Key in the dict returned by ``objective_fn`` whose
            value is returned to Optuna as the trial objective.
        n_trials: Target total number of completed trials. If the study
            already has this many, no new trials are run.
        random_seed: Seed for the TPESampler for reproducibility.
        top_n: Number of top-performing configs to save in YAML summary.
        eval_names: Ordered list of evaluation names used in this sweep.
        base_config_name: Name of the base configuration (for summary metadata).

    Returns:
        The completed Optuna study with all trial results.
    """
    study_paths.ensure_dirs()

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction=direction,
        study_name=study_paths.study_name,
        storage=study_paths.storage_url,
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(
        trial: optuna.trial.Trial,
        *,
        _sweep_space: dict[str, object] = sweep_space,
        _base_config: dict[str, object] = base_config,
        _primary_metric_key: str = primary_metric_key,
    ) -> float:
        trial_overrides = suggest_nested_params(trial, _sweep_space)
        merged_config = deep_merge_dicts(_base_config, trial_overrides)

        trial.set_user_attr(OptunaUserAttrs.CONFIG_OVERRIDES, trial_overrides)
        trial.set_user_attr(OptunaUserAttrs.MERGED_CONFIG, copy.deepcopy(merged_config))

        try:
            metrics = objective_fn(trial, merged_config)
            trial.set_user_attr(OptunaUserAttrs.EVAL_METRICS, metrics)
            return metrics[_primary_metric_key]

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

    max_trials_cb = optuna._callbacks.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
    study.optimize(objective, callbacks=[max_trials_cb])

    SweepTrialManager.summarize_top_n(
        study,
        direction=direction,
        top_n=top_n,
        eval_names=eval_names,
        base_cfg=base_config,
        base_config_name=base_config_name,
        output_dir=study_paths.sweep_results_dir,
        study_storage=study_paths.storage_path,
    )

    return study


def run_attack_grid_configs(
    attack_name: AttackName,
    config_grid: dict[str, dict[str, object]],
    pretrained_model_path: str,
    output_base_dir: Path,
    random_seed: int,
    eval_names: list[EvalName] | None = None,
    cleanup_checkpoints: bool = True,
) -> dict[str, DataFrame[EvaluationSchema]]:
    """Run an attack across all configs from a grid YAML.

    For each ``config_name`` in ``config_grid``:
      1. Inject ``input_checkpoint_path``, ``out_dir`` (``output_base_dir/config_name``),
         and ``random_seed`` into the config dict.
      2. If ``eval_names`` is provided, override evals in the config.
      3. Instantiate the attack from ``ATTACKS_REGISTRY`` and run ``attacker.benchmark()``.
      4. If ``cleanup_checkpoints`` is ``True``, call ``attacker.delete_output_checkpoint()``.
      5. Collect the results DataFrame.

    Args:
        attack_name: Name of the attack to run (must be registered in ``ATTACKS_REGISTRY``).
        config_grid: Mapping of config names to their configuration dictionaries,
            as loaded from a ``grid.yaml`` file.
        pretrained_model_path: Path to the pretrained model or checkpoint to attack.
        output_base_dir: Base directory for results. Each config gets a subdirectory
            at ``output_base_dir / config_name``.
        random_seed: Random seed injected into every config for reproducibility.
        eval_names: Optional list of eval names to override in each config.
            If ``None``, the evals specified in the grid config are used as-is.
        cleanup_checkpoints: Whether to delete output checkpoints after evaluation
            to save disk space (default: ``True``).

    Returns:
        Dictionary mapping config names to their evaluation results DataFrames.
    """
    attack_config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]
    results: dict[str, DataFrame[EvaluationSchema]] = {}

    for config_name, attack_config_dict in config_grid.items():
        config_dict = copy.deepcopy(attack_config_dict)
        config_dict[ConfigKeys.RANDOM_SEED] = random_seed

        if eval_names is not None:
            config_dict[ConfigKeys.EVALS] = [str(e) for e in eval_names]

        attacker: TamperAttack[TamperAttackConfig] = attack_cls(
            attack_config=attack_config_cls.from_dict(
                data={
                    "input_checkpoint_path": pretrained_model_path,
                    "out_dir": Path(output_base_dir, config_name),
                    **config_dict,
                }
            )
        )

        results_df: DataFrame[EvaluationSchema] = attacker.benchmark()
        results[config_name] = results_df

        if cleanup_checkpoints:
            attacker.delete_output_checkpoint()

    return results
