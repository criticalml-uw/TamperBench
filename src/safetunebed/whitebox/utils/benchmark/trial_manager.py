"""Trial management utilities for Optuna sweeps."""

# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.attacks.registry import ATTACKS_REGISTRY
from safetunebed.whitebox.evals.registry import EVALS_REGISTRY
from safetunebed.whitebox.utils import AttackName, OptimizationDirection
from safetunebed.whitebox.utils.benchmark.io import (
    deep_merge_dicts,
    dict_to_yaml,
    nest_dotted_params,
)
from safetunebed.whitebox.utils.benchmark.path_generation import get_trial_dir_name
from safetunebed.whitebox.utils.names import (
    ConfigKeys,
    EvalName,
    OptunaPaths,
    OptunaUserAttrs,
    TrialSummaryKeys,
)

if TYPE_CHECKING:
    import optuna
    from pandera.typing.polars import DataFrame

    from safetunebed.whitebox.evals.output_schema import EvaluationSchema


class SweepTrialManager:
    """Static methods for managing Optuna sweep trials.

    Provides utilities for running individual trials, sorting completed trials,
    and saving trial summaries. All methods are static since this class serves
    as a namespace for trial-related operations.

    Example:
        >>> from safetunebed.whitebox.utils.benchmark import SweepTrialManager
        >>> # Run a trial
        >>> results = SweepTrialManager.run_trial(
        ...     attack_name=AttackName.LORA_FINETUNE,
        ...     attack_config_dict=config,
        ...     eval_names=[EvalName.STRONG_REJECT],
        ...     pretrained_model_path="model_path",
        ...     attack_results_dir=Path("results/model/lora_finetune/optuna_single"),
        ...     trial_number=0,
        ... )
        >>> # Sort completed trials
        >>> sorted_trials = SweepTrialManager.sorted_completed_trials(study, "maximize")
    """

    @staticmethod
    def run_trial(
        attack_name: AttackName,
        attack_config_dict: dict[str, object],
        eval_names: list[EvalName],
        pretrained_model_path: str,
        attack_results_dir: Path,
        trial_number: int,
        random_seed: int = 42,
    ) -> dict[str, float]:
        """Run a single trial and return objective values keyed by eval name.

        Executes a complete attack benchmark cycle: instantiates the attack,
        runs the benchmark, collects evaluation metrics, and cleans up the
        output checkpoint.

        Args:
            attack_name: Name of the attack to run.
            attack_config_dict: Configuration dictionary for the attack.
            eval_names: List of evaluation names to compute metrics for.
            pretrained_model_path: Path to the pretrained model checkpoint.
            attack_results_dir: Directory for this attack's sweep results (e.g., from StudyPaths).
            trial_number: Optuna trial number for directory naming.
            random_seed: Random seed for reproducibility (default: 42).

        Returns:
            Dictionary mapping evaluation names to their objective values.

        Example:
            >>> results = SweepTrialManager.run_trial(
            ...     attack_name=AttackName.LORA_FINETUNE,
            ...     attack_config_dict={"learning_rate": 0.001},
            ...     eval_names=[EvalName.STRONG_REJECT],
            ...     pretrained_model_path="/path/to/model",
            ...     attack_results_dir=Path("results/model/lora_finetune/optuna_single"),
            ...     trial_number=5,
            ...     random_seed=42,
            ... )
            >>> results["strong_reject"]
            0.85
        """
        attack_config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]

        attack_config_data = copy.deepcopy(attack_config_dict)
        attack_config_data.update(
            {
                "input_checkpoint_path": pretrained_model_path,
                "out_dir": attack_results_dir / get_trial_dir_name(trial_number),
                ConfigKeys.RANDOM_SEED: random_seed,
            }
        )

        attacker: TamperAttack[TamperAttackConfig] = attack_cls(
            attack_config=attack_config_cls.from_dict(data=attack_config_data)
        )

        results_dataframe: DataFrame[EvaluationSchema] = attacker.benchmark()

        objective_values: dict[str, float] = {}
        for eval_name in eval_names:
            eval_key = str(eval_name)
            value = EVALS_REGISTRY[eval_name].load_result_objective(results_dataframe)
            objective_values[eval_key] = float(value)

        attacker.delete_output_checkpoint()

        return objective_values

    @staticmethod
    def sorted_completed_trials(study: optuna.study.Study, direction: str) -> list[optuna.trial.FrozenTrial]:
        """Return completed trials sorted best-first for single-objective optimization.

        Filters study trials to only include those that completed successfully
        with valid values, then sorts them according to the optimization direction.

        Args:
            study: Optuna study containing the trials.
            direction: Optimization direction, either "maximize" or "minimize".

        Returns:
            List of completed FrozenTrial objects sorted from best to worst.

        Example:
            >>> study = optuna.create_study(direction="maximize")
            >>> # ... run optimization ...
            >>> sorted_trials = SweepTrialManager.sorted_completed_trials(study, "maximize")
            >>> best_trial = sorted_trials[0]
        """
        import optuna

        completed_trials = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None  # pyright: ignore[reportAny]
        ]

        def sort_key(trial: optuna.trial.FrozenTrial) -> float:
            values: list[float] = trial.values  # pyright: ignore[reportAny]
            return -values[0] if direction == OptimizationDirection.MAXIMIZE else values[0]

        return sorted(completed_trials, key=sort_key)

    @staticmethod
    def build_summaries(
        top_trials: list[optuna.trial.FrozenTrial],
        eval_names: list[EvalName],
        base_cfg: Mapping[str, object],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Build best configs and trial summaries from top-ranked Optuna trials.

        Extracts merged configurations and summary information from completed Optuna
        trials. If a trial has ``merged_config`` in user_attrs, it is used directly;
        otherwise, the config is reconstructed by merging base_cfg with trial params.

        Args:
            top_trials: List of FrozenTrial objects sorted best-first.
            eval_names: List of evaluation names used in the sweep.
            base_cfg: Base configuration dictionary for fallback merging.

        Returns:
            A tuple of ``(best_configs, top_trial_summaries)`` where:

            - ``best_configs``: Dict mapping ``"top_1"``, ``"top_2"``, etc. to merged configs.
            - ``top_trial_summaries``: List of summary dicts containing rank, trial_number,
              values, eval_names, optuna_values, and params for each trial.

        Example:
            >>> sorted_trials = SweepTrialManager.sorted_completed_trials(study, "maximize")
            >>> top_trials = sorted_trials[:3]
            >>> best_configs, summaries = SweepTrialManager.build_summaries(
            ...     top_trials, [EvalName.STRONG_REJECT], base_attack_config
            ... )
            >>> best_configs["top_1"]["learning_rate"]
            0.0001
        """
        best_configs: dict[str, object] = {}
        summaries: list[dict[str, object]] = []

        for rank, trial in enumerate(top_trials, start=1):
            overrides: object = trial.user_attrs.get(OptunaUserAttrs.CONFIG_OVERRIDES)
            merged_config: object = trial.user_attrs.get(OptunaUserAttrs.MERGED_CONFIG)

            if merged_config is None:
                merged_config = deep_merge_dicts(
                    base_cfg,
                    nest_dotted_params(trial.params),
                )
                merged_config[ConfigKeys.EVALS] = [str(e) for e in eval_names]

            best_configs[f"top_{rank}"] = merged_config

            eval_metrics: object = trial.user_attrs.get(OptunaUserAttrs.EVAL_METRICS)

            if not isinstance(eval_metrics, dict):
                primary_eval_key = str(eval_names[0])
                trial_values: list[float] = trial.values  # pyright: ignore[reportAny]
                eval_metrics = {primary_eval_key: float(trial_values[0]) if trial_values else float("nan")}

            values_list: list[float] = [
                float(eval_metrics.get(str(e), float("nan")))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                for e in eval_names
            ]

            trial_values_for_summary: list[float] = trial.values  # pyright: ignore[reportAny]

            summaries.append(
                {
                    TrialSummaryKeys.RANK: rank,
                    TrialSummaryKeys.TRIAL_NUMBER: trial.number,
                    TrialSummaryKeys.VALUES: values_list,
                    TrialSummaryKeys.EVAL_NAMES: [str(e) for e in eval_names],
                    TrialSummaryKeys.OPTUNA_VALUES: trial_values_for_summary,
                    TrialSummaryKeys.PARAMS: overrides if overrides is not None else nest_dotted_params(trial.params),
                }
            )

        return best_configs, summaries

    @staticmethod
    def _save_summary(
        output_path: Path,
        *,
        base_config: str,
        directions: list[str],
        study_storage: Path,
        top_trials: list[dict[str, object]],
    ) -> None:
        """Save top trial summaries to a JSON file.

        Writes a structured JSON file containing metadata about the sweep
        and detailed information about the top-performing trials.

        Args:
            output_path: Path where the JSON file should be written.
            base_config: Name of the base configuration used for the sweep.
            directions: List of optimization directions for each objective.
            study_storage: Path to the Optuna study storage database.
            top_trials: List of trial summary dictionaries from build_summaries.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_config": base_config,
            "directions": directions,
            "study_storage": str(study_storage),
            "top_trials": top_trials,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    @staticmethod
    def summarize_top_n(
        study: optuna.study.Study,
        *,
        direction: str,
        top_n: int,
        eval_names: list[EvalName],
        base_cfg: Mapping[str, object],
        base_config_name: str,
        output_dir: Path,
        study_storage: Path,
    ) -> None:
        """Summarize and save the top N trials from an Optuna study.

        Combines the workflow of sorting completed trials, building summaries,
        and saving both YAML configs and JSON summaries to disk.

        Args:
            study: Completed Optuna study to summarize.
            direction: Optimization direction, either "maximize" or "minimize".
            top_n: Number of top-performing trials to save.
            eval_names: List of evaluation names used in the sweep.
            base_cfg: Base configuration dictionary used for the sweep.
            base_config_name: Name of the base configuration (for metadata).
            output_dir: Directory to save output files.
            study_storage: Path to the Optuna study storage database.

        Example:
            >>> SweepTrialManager.summarize_top_n(
            ...     study,
            ...     direction="maximize",
            ...     top_n=3,
            ...     eval_names=[EvalName.STRONG_REJECT],
            ...     base_cfg=base_attack_config,
            ...     base_config_name="base",
            ...     output_dir=Path("results/lora_finetune/optuna_single"),
            ...     study_storage=Path("results/study.db"),
            ... )
        """
        sorted_trials = SweepTrialManager.sorted_completed_trials(study, direction)
        top_trials = sorted_trials[: max(top_n, 0)]

        best_configs, top_trial_summaries = SweepTrialManager.build_summaries(top_trials, eval_names, base_cfg)

        dict_to_yaml(output_dir / OptunaPaths.BEST_YAML, best_configs)
        SweepTrialManager._save_summary(
            output_dir / OptunaPaths.BEST_JSON,
            base_config=base_config_name,
            directions=[direction],
            study_storage=study_storage,
            top_trials=top_trial_summaries,
        )
