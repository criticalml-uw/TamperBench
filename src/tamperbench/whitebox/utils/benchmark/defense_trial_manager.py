"""Defense trial management: defend -> eval -> attack -> eval -> cleanup."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from tamperbench.utils import get_repo_root
from tamperbench.whitebox.defenses.registry import DEFENSES_REGISTRY
from tamperbench.whitebox.evals.registry import EVAL_CONFIG_REGISTRY, EVALS_REGISTRY
from tamperbench.whitebox.utils.benchmark.attack_aggregation import aggregate_metrics
from tamperbench.whitebox.utils.benchmark.config import (
    AttackSweepConfig,
    load_attack_base_config,
)
from tamperbench.whitebox.utils.benchmark.defense_config import CapabilityGuard
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.benchmark.path_generation import (
    StudyPaths,
    get_trial_dir_name,
)
from tamperbench.whitebox.utils.benchmark.runners import (
    run_attack_grid_configs,
    run_optuna_sweep,
)
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import (
    AggregationConfigKeys,
    BestCheckpointKeys,
    ConfigKeys,
    ConfigPath,
    DefenseMetricPrefix,
    DefenseName,
    DefenseTrialDirs,
    EvalName,
    OptimizationDirection,
    OptunaPaths,
)

if TYPE_CHECKING:
    import optuna

    from tamperbench.whitebox.utils.benchmark.defense_config import PostDefenseAttackSpec

logger = logging.getLogger(__name__)

WHITEBOX_ATTACK_CONFIG_DIR = get_repo_root() / Path("configs", "whitebox", "attacks")


class DefenseSweepTrialManager:
    """Static methods for managing defense sweep trials.

    Provides utilities for running the full defense trial pipeline:
    defend -> evaluate defense -> attack(s) -> evaluate post-attack -> cleanup.
    All methods are static since this class serves as a namespace for
    defense trial-related operations.

    Example:
        >>> from tamperbench.whitebox.utils.benchmark.defense_trial_manager import DefenseSweepTrialManager
        >>> metrics = DefenseSweepTrialManager.run_trial(
        ...     defense_name=DefenseName.CRL,
        ...     defense_config_dict=config,
        ...     defense_eval_names=[EvalName.STRONG_REJECT],
        ...     post_attack_eval_names=[EvalName.STRONG_REJECT],
        ...     model_config_dict={"template": "plain", "max_generation_length": 1024, "inference_batch_size": 16},
        ...     attacks=[PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE)],
        ...     pretrained_model_path="model_path",
        ...     defense_results_dir=Path("results/model/crl/optuna_single"),
        ...     trial_number=0,
        ...     random_seed=42,
        ...     model_alias="llama3_8b",
        ... )
    """

    @staticmethod
    def run_trial(
        defense_name: DefenseName,
        defense_config_dict: dict[str, object],
        defense_eval_names: list[EvalName],
        post_attack_eval_names: list[EvalName],
        model_config_dict: dict[str, object],
        attacks: list[PostDefenseAttackSpec],
        pretrained_model_path: str,
        defense_results_dir: Path,
        trial_number: int,
        random_seed: int,
        attack_configs_dir: Path | None = None,
        model_alias: str = "",
        capability_guards: dict[EvalName, CapabilityGuard] | None = None,
        capability_baselines: dict[str, float] | None = None,
        cleanup_checkpoints: bool = True,
        best_checkpoint_dir: Path | None = None,
        primary_metric_key: str | None = None,
        optimization_direction: OptimizationDirection | None = None,
    ) -> dict[str, float]:
        """Run a single defense trial.

        Executes the full pipeline: build defense config, run defense, evaluate
        the defended checkpoint, run post-defense attacks, collect metrics,
        optionally clean up the defended checkpoint, and save aggregated trial
        results.

        If ``capability_guards`` and ``capability_baselines`` are provided,
        the defended checkpoint is checked after defense evaluation.  When any
        guarded metric drops below ``baseline * min_retention``, the expensive
        post-defense attack phase is skipped and penalty values are returned
        for all post-attack metrics so that Optuna learns to avoid that region.

        Args:
            defense_name: Name of the defense to run from the defense registry.
            defense_config_dict: Configuration dictionary for the defense, including
                meta-fields that will be stripped before passing to the defense.
            defense_eval_names: Evaluations to run on the defended checkpoint
                (pre-attack, post-defense).
            post_attack_eval_names: Evaluations to run on post-attack checkpoints.
            model_config_dict: Raw model configuration dictionary for inference.
            attacks: List of attack specifications to stress-test the defense.
            pretrained_model_path: Path to the pretrained model checkpoint.
            defense_results_dir: Directory for this defense's sweep results
                (e.g., ``results/model/crl/optuna_single/``).
            trial_number: Trial number for directory naming.
            random_seed: Random seed for reproducibility.
            attack_configs_dir: Optional root directory for attack configs.
                Falls back to the repository default if not specified.
            model_alias: Alias for the model, used for inner attack sweep
                StudyPaths to keep consistent directory structure.
            capability_guards: Optional mapping of eval names to retention
                thresholds.  Evaluated after the defense step.
            capability_baselines: Base model scores for the guarded metrics,
                obtained once at the start of the sweep via
                ``evaluate_baselines()``.
            cleanup_checkpoints: Whether to delete the defended checkpoint
                after evaluation and attacks (default: ``True``).  Set to
                ``False`` to keep defended checkpoints for inspection.
            best_checkpoint_dir: If provided, the best defended checkpoint
                is saved here, replacing the previous best when a better
                trial is found. Requires ``primary_metric_key`` and
                ``optimization_direction``.
            primary_metric_key: Metric key to compare for best checkpoint.
            optimization_direction: Direction for best checkpoint comparison.

        Returns:
            Flat dictionary mapping prefixed metric names to their values.
            Keys use dot-separated prefixes, e.g.
            ``{"defense.strong_reject": 0.12, "post_attack.lora_finetune.strong_reject": 0.45}``.
        """
        trial_dir = defense_results_dir / get_trial_dir_name(trial_number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build defense config, stripping meta-fields
        clean_config = DefenseSweepTrialManager._strip_meta_fields(copy.deepcopy(defense_config_dict))

        defended_checkpoint_dir = trial_dir / DefenseTrialDirs.DEFENDED_MODEL
        clean_config["input_checkpoint_path"] = Path(pretrained_model_path)
        clean_config["output_checkpoint_path"] = defended_checkpoint_dir

        defense_config_cls, defense_cls = DEFENSES_REGISTRY[defense_name]
        defense_config = defense_config_cls.from_dict(clean_config)
        defense = defense_cls(defense_config=defense_config)

        # 2. Run defense
        defended_checkpoint = str(defense.run_defense())
        logger.info("Defense checkpoint created at: %s", defended_checkpoint)

        model_config = ModelConfig.from_dict(dict(model_config_dict))
        all_metrics: dict[str, float] = {}

        # 3. Evaluate defended checkpoint with defense_evals
        defense_eval_dir = trial_dir / DefenseTrialDirs.DEFENSE_EVAL
        defense_metrics = DefenseSweepTrialManager.evaluate_checkpoint(
            checkpoint_path=defended_checkpoint,
            eval_names=defense_eval_names,
            model_config=model_config,
            out_dir=defense_eval_dir,
        )
        all_metrics.update(DefenseSweepTrialManager._prefix_metrics(DefenseMetricPrefix.DEFENSE, defense_metrics))

        # 3b. Check capability guards — skip attacks if defense destroyed utility
        guard_violation = DefenseSweepTrialManager._check_capability_guards(
            defense_metrics=defense_metrics,
            capability_guards=capability_guards,
            capability_baselines=capability_baselines,
        )
        if guard_violation is not None:
            logger.warning(
                "Trial %d: capability guard violated — %s. Skipping attack phase.",
                trial_number,
                guard_violation,
            )
            # Fill post-attack metrics with penalty values (worst-case for defender)
            for attack_spec in attacks:
                penalty_metrics = DefenseSweepTrialManager._penalty_metrics(post_attack_eval_names)
                all_metrics.update(
                    DefenseSweepTrialManager._prefix_metrics(
                        f"{DefenseMetricPrefix.POST_ATTACK}.{attack_spec.name}", penalty_metrics
                    )
                )

            # Cleanup and save
            if cleanup_checkpoints and Path(defended_checkpoint).exists():
                shutil.rmtree(defended_checkpoint)
                logger.info("Cleaned up defended checkpoint: %s", defended_checkpoint)

            trial_results_path = trial_dir / DefenseTrialDirs.TRIAL_RESULTS
            DefenseSweepTrialManager._save_trial_results(trial_results_path, all_metrics)
            return all_metrics

        # 4. For each attack spec, run the attack against the defended checkpoint
        post_attack_dir = trial_dir / DefenseTrialDirs.POST_ATTACK
        for attack_spec in attacks:
            attack_out_dir = post_attack_dir / str(attack_spec.name)
            attack_out_dir.mkdir(parents=True, exist_ok=True)

            if attack_spec.mode == "grid":
                attack_metrics = DefenseSweepTrialManager.run_attack_grid(
                    attack_spec=attack_spec,
                    defended_checkpoint=defended_checkpoint,
                    post_attack_eval_names=post_attack_eval_names,
                    model_config_dict=model_config_dict,
                    attack_out_dir=attack_out_dir,
                    random_seed=random_seed,
                    attack_configs_dir=attack_configs_dir,
                    defense_metrics=defense_metrics,
                )
            else:
                attack_metrics = DefenseSweepTrialManager.run_attack_sweep(
                    attack_spec=attack_spec,
                    defended_checkpoint=defended_checkpoint,
                    post_attack_eval_names=post_attack_eval_names,
                    model_config_dict=model_config_dict,
                    attack_out_dir=attack_out_dir,
                    random_seed=random_seed,
                    attack_configs_dir=attack_configs_dir,
                    model_alias=model_alias,
                )

            all_metrics.update(
                DefenseSweepTrialManager._prefix_metrics(
                    f"{DefenseMetricPrefix.POST_ATTACK}.{attack_spec.name}", attack_metrics
                )
            )

        # 5. Save best checkpoint if this trial is the new best
        if best_checkpoint_dir is not None and primary_metric_key is not None and optimization_direction is not None:
            DefenseSweepTrialManager._maybe_save_best_checkpoint(
                defended_checkpoint=defended_checkpoint,
                best_checkpoint_dir=best_checkpoint_dir,
                primary_metric_key=primary_metric_key,
                optimization_direction=optimization_direction,
                all_metrics=all_metrics,
                trial_number=trial_number,
            )

        # 6. Optionally delete defended checkpoint
        if cleanup_checkpoints and Path(defended_checkpoint).exists():
            shutil.rmtree(defended_checkpoint)
            logger.info("Cleaned up defended checkpoint: %s", defended_checkpoint)

        # 7. Save trial_results.json
        trial_results_path = trial_dir / DefenseTrialDirs.TRIAL_RESULTS
        DefenseSweepTrialManager._save_trial_results(trial_results_path, all_metrics)

        return all_metrics

    @staticmethod
    def evaluate_checkpoint(
        checkpoint_path: str,
        eval_names: list[EvalName],
        model_config: ModelConfig,
        out_dir: Path,
    ) -> dict[str, float]:
        """Run evaluations on a checkpoint without any attack wrapper.

        Constructs evaluation configs and runs each registered evaluation
        on the given checkpoint, collecting the objective metric from each.

        Args:
            checkpoint_path: Path to the model checkpoint to evaluate.
            eval_names: List of evaluation names to run.
            model_config: Model configuration for inference.
            out_dir: Directory where evaluation results will be stored.

        Returns:
            Dictionary mapping evaluation name strings to their objective values.

        Example:
            >>> metrics = DefenseSweepTrialManager.evaluate_checkpoint(
            ...     checkpoint_path="/path/to/defended_model",
            ...     eval_names=[EvalName.STRONG_REJECT, EvalName.MMLU_PRO_VAL],
            ...     model_config=model_config,
            ...     out_dir=Path("results/defense_eval"),
            ... )
            >>> metrics["strong_reject"]
            0.12
        """
        results: dict[str, float] = {}

        for eval_name in eval_names:
            eval_cls = EVALS_REGISTRY[eval_name]
            config_cls = EVAL_CONFIG_REGISTRY[eval_name]
            eval_config = config_cls(
                model_checkpoint=checkpoint_path,
                out_dir=str(out_dir),
                model_config=model_config,
            )
            evaluation = eval_cls(eval_config)
            result_df = evaluation.run_evaluation()
            value = eval_cls.load_result_objective(result_df)
            results[str(eval_name)] = float(value)

        return results

    @staticmethod
    def run_attack_grid(
        attack_spec: PostDefenseAttackSpec,
        defended_checkpoint: str,
        post_attack_eval_names: list[EvalName],
        model_config_dict: dict[str, object],
        attack_out_dir: Path,
        random_seed: int,
        attack_configs_dir: Path | None = None,
        defense_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Run attack in grid mode against a defended checkpoint.

        Loads the attack's grid.yaml, optionally filters to a single config,
        injects model config and evals, and runs all configs. Returns
        aggregated metrics using the method specified in ``attack_spec.aggregation``.

        Args:
            attack_spec: Attack specification defining name, mode, and config options.
            defended_checkpoint: Path to the defended model checkpoint.
            post_attack_eval_names: Evaluations to run after each attack config.
            model_config_dict: Raw model configuration dictionary for inference.
            attack_out_dir: Output directory for attack results.
            random_seed: Random seed for reproducibility.
            attack_configs_dir: Root directory for attack configs. Falls back to
                the repository default if not specified.
            defense_metrics: Metrics from the defended checkpoint (pre-attack).
                Required when using ``bounded_worst_case`` aggregation.

        Returns:
            Dictionary mapping evaluation name strings to aggregated metric values.
        """
        configs_dir = _resolve_attack_configs_dir(attack_spec.configs_dir, attack_configs_dir)
        grid_path = Path(configs_dir, str(attack_spec.name), ConfigPath.GRID_YAML)
        config_grid: dict[str, dict[str, object]] = {}

        raw_grid = yaml_to_dict(grid_path)
        if attack_spec.config_name is not None:
            raw_value = raw_grid[attack_spec.config_name]
            if isinstance(raw_value, dict):
                config_grid[attack_spec.config_name] = raw_value
        else:
            for key, value in raw_grid.items():
                if isinstance(value, dict):
                    config_grid[key] = value

        # Inject model_config and evals into each config dict
        for config_dict in config_grid.values():
            config_dict[ConfigKeys.MODEL_CONFIG] = copy.deepcopy(model_config_dict)
            config_dict[ConfigKeys.EVALS] = [str(e) for e in post_attack_eval_names]

        results = run_attack_grid_configs(
            attack_name=attack_spec.name,
            config_grid=config_grid,
            pretrained_model_path=defended_checkpoint,
            output_base_dir=attack_out_dir,
            random_seed=random_seed,
            eval_names=post_attack_eval_names,
            cleanup_checkpoints=True,
        )

        # Extract objective values per config
        all_config_results: dict[str, dict[str, float]] = {}
        for config_name, results_df in results.items():
            config_metrics: dict[str, float] = {}
            for eval_name in post_attack_eval_names:
                eval_cls = EVALS_REGISTRY[eval_name]
                config_metrics[str(eval_name)] = float(eval_cls.load_result_objective(results_df))
            all_config_results[config_name] = config_metrics

        aggregation = attack_spec.aggregation
        return aggregate_metrics(
            all_config_results=all_config_results,
            eval_names=post_attack_eval_names,
            method=aggregation.method,
            defense_metrics=defense_metrics,
            **{
                AggregationConfigKeys.N: aggregation.n,
                AggregationConfigKeys.BOUND_EVAL: str(aggregation.bound_eval) if aggregation.bound_eval else None,
                AggregationConfigKeys.BOUND_MIN_RETENTION: aggregation.bound_min_retention,
            },
        )

    @staticmethod
    def run_attack_sweep(
        attack_spec: PostDefenseAttackSpec,
        defended_checkpoint: str,
        post_attack_eval_names: list[EvalName],
        model_config_dict: dict[str, object],
        attack_out_dir: Path,
        random_seed: int,
        attack_configs_dir: Path | None = None,
        model_alias: str = "",
    ) -> dict[str, float]:
        """Run attack in sweep mode using an inner Optuna optimization.

        Loads the attack's sweep config and base config, creates an inner
        Optuna study, and runs the sweep to find the most effective attack
        hyperparameters. Returns the best trial's metrics.

        Args:
            attack_spec: Attack specification defining name, mode, and sweep options.
            defended_checkpoint: Path to the defended model checkpoint.
            post_attack_eval_names: Evaluations to run after each attack trial.
            model_config_dict: Raw model configuration dictionary for inference.
            attack_out_dir: Output directory for inner sweep results.
            random_seed: Random seed for reproducibility.
            attack_configs_dir: Root directory for attack configs. Falls back to
                the repository default if not specified.
            model_alias: Alias for the model, used for inner StudyPaths.

        Returns:
            Dictionary mapping evaluation name strings to the best trial's
            metric values from the attacker's perspective.
        """
        configs_dir = _resolve_attack_configs_dir(attack_spec.configs_dir, attack_configs_dir)

        # Load base config and sweep config for the attack
        base_config_name, base_attack_config = load_attack_base_config(configs_dir, attack_spec.name)
        sweep_config = AttackSweepConfig.from_yaml(
            configs_dir, attack_spec.name, sweep_filename=ConfigPath.SINGLE_OBJECTIVE_SWEEP
        )

        # Override evals and inject model_config into the base config
        base_attack_config[ConfigKeys.EVALS] = [str(e) for e in post_attack_eval_names]
        base_attack_config[ConfigKeys.MODEL_CONFIG] = copy.deepcopy(model_config_dict)

        # Determine attacker direction from the primary eval
        primary_eval_name = post_attack_eval_names[0]
        primary_eval_cls = EVALS_REGISTRY[primary_eval_name]
        direction = primary_eval_cls.attacker_direction.value

        # Create inner StudyPaths for the attack sweep
        inner_study_paths = StudyPaths(
            results_dir=attack_out_dir,
            model_alias=model_alias,
            attack_name=attack_spec.name,
            sweep_subdir=OptunaPaths.OPTUNA_SINGLE_DIR,
        )

        # Build inner objective function
        def inner_objective(trial: optuna.trial.Trial, merged_config: dict[str, object]) -> dict[str, float]:
            merged_config[ConfigKeys.EVALS] = [str(e) for e in post_attack_eval_names]
            merged_config[ConfigKeys.RANDOM_SEED] = random_seed

            return SweepTrialManager.run_trial(
                attack_name=attack_spec.name,
                attack_config_dict=merged_config,
                eval_names=post_attack_eval_names,
                pretrained_model_path=defended_checkpoint,
                attack_results_dir=inner_study_paths.sweep_results_dir,
                trial_number=trial.number,
                random_seed=random_seed,
            )

        study = run_optuna_sweep(
            study_paths=inner_study_paths,
            sweep_space=sweep_config.sweep,
            base_config=base_attack_config,
            direction=direction,
            objective_fn=inner_objective,
            primary_metric_key=str(primary_eval_name),
            n_trials=attack_spec.n_trials,
            random_seed=random_seed,
            top_n=1,
            eval_names=post_attack_eval_names,
            base_config_name=base_config_name,
        )

        # Return best trial's metrics from the inner sweep
        sorted_trials = SweepTrialManager.sorted_completed_trials(study, direction)
        if not sorted_trials:
            logger.warning("Inner attack sweep for %s produced no completed trials.", attack_spec.name)
            return {str(e): float("nan") for e in post_attack_eval_names}

        best_trial = sorted_trials[0]
        eval_metrics: object = best_trial.user_attrs.get("eval_metrics")
        if isinstance(eval_metrics, dict):
            return {str(k): float(v) for k, v in eval_metrics.items()}

        # Fallback: return the primary objective value
        best_values: list[float] = best_trial.values
        return {str(primary_eval_name): float(best_values[0]) if best_values else float("nan")}

    @staticmethod
    def _strip_meta_fields(config_dict: dict[str, object]) -> dict[str, object]:
        """Remove meta-fields that should not be passed to AlignmentDefenseConfig.

        Strips keys used for orchestration (evaluations, attacks, model config)
        that are not part of the defense's own hyperparameters.

        Args:
            config_dict: Raw defense configuration dictionary potentially
                containing meta-fields.

        Returns:
            A copy of the dictionary with meta-fields removed.
        """
        meta_keys = {
            str(ConfigKeys.DEFENSE_EVALS),
            str(ConfigKeys.POST_ATTACK_EVALS),
            str(ConfigKeys.ATTACKS),
            str(ConfigKeys.MODEL_CONFIG),
            str(ConfigKeys.CAPABILITY_GUARDS),
        }
        return {k: v for k, v in config_dict.items() if k not in meta_keys}

    @staticmethod
    def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
        """Add a dot-separated prefix to metric keys.

        Args:
            prefix: Prefix string to prepend (e.g., ``"defense"`` or
                ``"post_attack.lora_finetune"``).
            metrics: Dictionary of metric name to value mappings.

        Returns:
            New dictionary with prefixed keys, e.g.
            ``{"defense.strong_reject": 0.12}``.
        """
        return {f"{prefix}.{key}": value for key, value in metrics.items()}

    @staticmethod
    def _check_capability_guards(
        defense_metrics: dict[str, float],
        capability_guards: dict[EvalName, CapabilityGuard] | None,
        capability_baselines: dict[str, float] | None,
    ) -> str | None:
        """Check whether the defended model violates any capability guard.

        Args:
            defense_metrics: Metrics from evaluating the defended checkpoint,
                keyed by bare eval name (e.g. ``"mmlu_pro_val"``).
            capability_guards: Per-metric retention thresholds.
            capability_baselines: Base model scores for the guarded metrics.

        Returns:
            A human-readable violation message, or ``None`` if all guards pass.
        """
        if not capability_guards or not capability_baselines:
            return None

        for eval_name, guard in capability_guards.items():
            eval_key = str(eval_name)
            defended_value = defense_metrics.get(eval_key)
            baseline_value = capability_baselines.get(eval_key)

            if defended_value is None or baseline_value is None:
                continue

            threshold = baseline_value * guard.min_retention
            if defended_value < threshold:
                return (
                    f"{eval_key} dropped to {defended_value:.4f} "
                    f"(baseline {baseline_value:.4f}, "
                    f"threshold {threshold:.4f} = {guard.min_retention:.0%} retention)"
                )

        return None

    @staticmethod
    def _penalty_metrics(eval_names: list[EvalName]) -> dict[str, float]:
        """Return worst-case penalty values for each eval from the defender's perspective.

        For safety evals where defender wants to minimize (e.g. strong_reject),
        returns 1.0 (maximum harm). For capability evals where defender wants
        to maximize (e.g. mmlu_pro_val), returns 0.0 (no capability).
        """
        penalties: dict[str, float] = {}
        for eval_name in eval_names:
            eval_cls = EVALS_REGISTRY[eval_name]
            if eval_cls.defender_direction == OptimizationDirection.MINIMIZE:
                # Defender wants low → penalty is high (worst for defender)
                penalties[str(eval_name)] = 1.0
            else:
                # Defender wants high → penalty is low (worst for defender)
                penalties[str(eval_name)] = 0.0
        return penalties

    @staticmethod
    def evaluate_baselines(
        pretrained_model_path: str,
        capability_guards: dict[EvalName, CapabilityGuard],
        model_config: ModelConfig,
        out_dir: Path,
    ) -> dict[str, float]:
        """Evaluate the base (undefended) model on guarded metrics.

        Called once at the start of a sweep to establish baseline capability
        scores that ``_check_capability_guards`` compares against.

        Args:
            pretrained_model_path: Path to the undefended model checkpoint.
            capability_guards: Guards whose eval names determine which
                evaluations to run.
            model_config: Model configuration for inference.
            out_dir: Directory for baseline evaluation results.

        Returns:
            Dictionary mapping eval name strings to baseline scores.
        """
        guard_eval_names = list(capability_guards.keys())
        if not guard_eval_names:
            return {}

        logger.info(
            "Evaluating base model on guarded metrics: %s",
            [str(e) for e in guard_eval_names],
        )
        baselines = DefenseSweepTrialManager.evaluate_checkpoint(
            checkpoint_path=pretrained_model_path,
            eval_names=guard_eval_names,
            model_config=model_config,
            out_dir=out_dir,
        )
        logger.info("Capability baselines: %s", baselines)
        return baselines

    @staticmethod
    def _maybe_save_best_checkpoint(
        defended_checkpoint: str,
        best_checkpoint_dir: Path,
        primary_metric_key: str,
        optimization_direction: OptimizationDirection,
        all_metrics: dict[str, float],
        trial_number: int,
    ) -> None:
        """Save the defended checkpoint if it has the best primary metric so far.

        Compares against the score in ``best_checkpoint_dir/best_score.json``.
        If this trial is better (or no previous best exists), replaces the
        checkpoint and updates the score file.
        """
        current_score = all_metrics.get(primary_metric_key)
        if current_score is None:
            return

        best_score_path = best_checkpoint_dir / DefenseTrialDirs.BEST_CHECKPOINT_SCORE
        is_best = False

        if best_score_path.exists():
            with open(best_score_path) as f:
                prev = json.load(f)
            prev_score = prev[BestCheckpointKeys.SCORE]
            if optimization_direction == OptimizationDirection.MINIMIZE:
                is_best = current_score < prev_score
            else:
                is_best = current_score > prev_score
        else:
            is_best = True

        if is_best:
            # Clear old checkpoint files (everything except best_score.json)
            if best_checkpoint_dir.exists():
                for item in best_checkpoint_dir.iterdir():
                    if item.name != DefenseTrialDirs.BEST_CHECKPOINT_SCORE:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(defended_checkpoint, best_checkpoint_dir, dirs_exist_ok=True)
            with open(best_score_path, "w") as f:
                json.dump(
                    {
                        BestCheckpointKeys.TRIAL: trial_number,
                        BestCheckpointKeys.SCORE: current_score,
                        BestCheckpointKeys.METRIC: primary_metric_key,
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "New best checkpoint (trial %d, %s=%.4f) saved to %s",
                trial_number, primary_metric_key, current_score, best_checkpoint_dir,
            )

    @staticmethod
    def _save_trial_results(path: Path, metrics: dict[str, float]) -> None:
        """Write aggregated metrics dictionary to a JSON file.

        Args:
            path: File path where the JSON should be written.
            metrics: Dictionary of metric names to values.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)


def _resolve_attack_configs_dir(
    spec_configs_dir: Path | None,
    fallback_configs_dir: Path | None,
) -> Path:
    """Resolve the attack configs directory from spec, fallback, or repo default.

    Args:
        spec_configs_dir: Directory from the attack spec (highest priority).
        fallback_configs_dir: Caller-provided fallback directory.

    Returns:
        Resolved path to the attack configs directory.
    """
    if spec_configs_dir is not None:
        return spec_configs_dir
    if fallback_configs_dir is not None:
        return fallback_configs_dir
    return WHITEBOX_ATTACK_CONFIG_DIR
