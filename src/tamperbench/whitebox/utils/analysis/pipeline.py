"""High-level analysis pipeline for epsilon-bounded sweep analysis.

Provides AnalysisPipeline to coordinate data collection, epsilon-bounded filtering,
heatmap building, and results saving.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from optuna.trial._frozen import FrozenTrial

from tamperbench.whitebox.utils.analysis.builders import HeatmapBuilder, HeatmapData
from tamperbench.whitebox.utils.analysis.collectors import OptunaCollector, StudyData
from tamperbench.whitebox.utils.analysis.config import AnalysisConfig
from tamperbench.whitebox.utils.analysis.filters import EpsilonBoundedFilter
from tamperbench.whitebox.utils.analysis.io import ensure_dir, write_json
from tamperbench.whitebox.utils.analysis.metrics import get_metric_value
from tamperbench.whitebox.utils.analysis.validation import ResultsValidator
from tamperbench.whitebox.utils.benchmark.path_generation import (
    get_aggregated_dir_name,
    get_trial_dir_name,
    get_trial_eval_dir_name,
)
from tamperbench.whitebox.utils.names import (
    AttackName,
    ConfigPath,
    EvalName,
    EvalPath,
    MetricName,
    OptunaPaths,
    OptunaUserAttrs,
)

LOGGER = logging.getLogger(__name__)


class AnalysisPipeline:
    """End-to-end analysis workflow orchestration."""

    def __init__(self, config: AnalysisConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Analysis configuration
        """
        self.config: AnalysisConfig = config
        self.collector: OptunaCollector = OptunaCollector(config)
        self.heatmap_builder: HeatmapBuilder = HeatmapBuilder(config)
        self.validator: ResultsValidator = ResultsValidator(config)

    def run_epsilon_bounded(
        self,
        parent_results_dir: Path,
        epsilons: list[float],
        n_trials: int,
        skip_aggregation: bool = False,
    ) -> None:
        """Run epsilon-bounded analysis pipeline.

        Filters trials by utility degradation constraints and selects the best attack
        (highest safety metric) within those constraints. For each epsilon, outputs
        heatmaps, visualizations, and trial artifacts.

        Args:
            parent_results_dir: Parent results directory containing model/attack subdirs
            epsilons: List of epsilon bounds (fraction of utility degradation allowed)
            n_trials: Number of chronological trials to consider per attack
            skip_aggregation: If True, skip aggregation step

        Example:
            >>> pipeline = AnalysisPipeline(config)
            >>> pipeline.run_epsilon_bounded(
            ...     Path("results/sweep_001"),
            ...     epsilons=[0.01, 0.03, 0.05],
            ...     n_trials=50,
            ... )
        """
        LOGGER.info(f"Running epsilon-bounded pipeline with epsilons={epsilons}, n_trials={n_trials}")

        self.validator.validate_directory(results_dir=parent_results_dir, require_study_db=True)

        epsilon_filter = EpsilonBoundedFilter(self.config.epsilon_bounded)
        baseline_attack = self.config.epsilon_bounded.baseline_attack

        all_data = self.collector.collect_all_models(parent_results_dir)
        baseline_utility_scores = self.get_baseline_utility_scores(all_data, baseline_attack)

        for epsilon in epsilons:
            self.process_single_epsilon(
                epsilon=epsilon,
                all_data=all_data,
                baseline_utility_scores=baseline_utility_scores,
                baseline_attack=baseline_attack,
                epsilon_filter=epsilon_filter,
                n_trials=n_trials,
                parent_results_dir=parent_results_dir,
                skip_aggregation=skip_aggregation,
            )

        LOGGER.info("Epsilon-bounded analysis complete")

    def process_single_epsilon(
        self,
        epsilon: float,
        all_data: dict[str, dict[AttackName, StudyData]],
        baseline_utility_scores: dict[str, float],
        baseline_attack: AttackName,
        epsilon_filter: EpsilonBoundedFilter,
        n_trials: int,
        parent_results_dir: Path,
        skip_aggregation: bool,
    ) -> None:
        """Filter trials, build heatmaps, and generate outputs for a single epsilon."""
        eps_str = f"eps{int(epsilon * 100):02d}"
        LOGGER.info(f"Processing epsilon={epsilon} ({eps_str})")

        output_dir = parent_results_dir / get_aggregated_dir_name(epsilon)
        ensure_dir(output_dir)

        if skip_aggregation:
            return

        heatmap_data, selected_trials_info = self.build_heatmap_matrices(
            all_data=all_data,
            baseline_utility_scores=baseline_utility_scores,
            baseline_attack=baseline_attack,
            epsilon=epsilon,
            epsilon_filter=epsilon_filter,
            n_trials=n_trials,
            parent_results_dir=parent_results_dir,
        )

        heatmap_json_path = output_dir / "heatmap_max_sr.json"
        write_json(heatmap_data.to_dict(), heatmap_json_path)

        self.copy_trial_artifacts(selected_trials_info, all_data, output_dir)

        LOGGER.info(f"Epsilon {eps_str} complete: {output_dir}")

    def get_baseline_utility_scores(
        self,
        all_data: dict[str, dict[AttackName, StudyData]],
        baseline_attack: AttackName,
    ) -> dict[str, float]:
        """Extract baseline utility scores for each model from baseline attack trials."""
        baseline_scores: dict[str, float] = {}
        utility_metric_name = MetricName(self.config.epsilon_bounded.utility_metric)

        for model_name, model_data in all_data.items():
            if baseline_attack not in model_data:
                raise ValueError(f"No baseline data found for {model_name}")

            baseline_study = model_data[baseline_attack].study
            completed_trials = [t for t in baseline_study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE]

            if not completed_trials:
                raise ValueError(f"No completed trials for {model_name} baseline")

            baseline_trial: FrozenTrial = completed_trials[0]
            baseline_utility = get_metric_value(baseline_trial, utility_metric_name, self.config)

            if baseline_utility is not None:
                baseline_scores[model_name] = baseline_utility
                LOGGER.debug(f"Baseline {utility_metric_name} for {model_name}: {baseline_utility}")

        return baseline_scores

    def build_heatmap_matrices(
        self,
        all_data: dict[str, dict[AttackName, StudyData]],
        baseline_utility_scores: dict[str, float],
        baseline_attack: AttackName,
        epsilon: float,
        epsilon_filter: EpsilonBoundedFilter,
        n_trials: int,
        parent_results_dir: Path,
    ) -> tuple[HeatmapData, dict[tuple[str, AttackName], dict[str, Any]]]:
        """Build heatmap matrices from epsilon-filtered trials, returning data and trial info."""
        harmfulness_metric_name = MetricName(self.config.epsilon_bounded.safety_metric)
        utility_metric_name = MetricName(self.config.epsilon_bounded.utility_metric)

        models = self.heatmap_builder.order_models(list(all_data.keys()))
        attacks = self.heatmap_builder.order_attacks(all_data)

        n_models = len(models)
        n_attacks = len(attacks)
        harmfulness_raw = np.full((n_models, n_attacks), np.nan)
        utility_raw = np.full((n_models, n_attacks), np.nan)

        selected_trials_info: dict[tuple[str, AttackName], dict[str, Any]] = {}

        for i, model in enumerate(models):
            baseline_utility = baseline_utility_scores.get(model)

            for j, attack in enumerate(attacks):
                if attack not in all_data.get(model, {}):
                    continue

                study_data = all_data[model][attack]

                selected_trial = self.select_trial_for_model_attack(
                    study_data=study_data,
                    attack=attack,
                    baseline_attack=baseline_attack,
                    baseline_utility=baseline_utility,
                    epsilon=epsilon,
                    n_trials=n_trials,
                    epsilon_filter=epsilon_filter,
                )

                if selected_trial is None:
                    continue

                # Track trial information for later artifact copying
                trial_number = selected_trial.number
                trial_dir = (
                    parent_results_dir
                    / model
                    / attack
                    / OptunaPaths.OPTUNA_SINGLE_DIR
                    / get_trial_dir_name(trial_number)
                )
                selected_trials_info[(model, attack)] = {
                    "trial_number": trial_number,
                    "trial_dir": trial_dir,
                }

                # Extract metrics for heatmap
                harmfulness_val, utility_val = self.extract_metrics_for_heatmap(
                    selected_trial=selected_trial,
                    attack=attack,
                    model=model,
                    all_data=all_data,
                    baseline_attack=baseline_attack,
                    harmfulness_metric_name=harmfulness_metric_name,
                    utility_metric_name=utility_metric_name,
                )

                if harmfulness_val is not None:
                    harmfulness_raw[i, j] = harmfulness_val
                if utility_val is not None:
                    utility_raw[i, j] = utility_val

        # Calculate deltas
        harmfulness_delta = self.heatmap_builder.calculate_deltas(harmfulness_raw, attacks, AttackName(baseline_attack))
        utility_delta = self.heatmap_builder.calculate_deltas(utility_raw, attacks, AttackName(baseline_attack))

        heatmap_data = HeatmapData(
            safety_raw=harmfulness_raw,
            safety_delta=harmfulness_delta,
            utility_raw=utility_raw,
            utility_delta=utility_delta,
            models=models,
            attacks=attacks,
            metadata={
                "method": "epsilon_bounded",
                "epsilon": epsilon,
                "n_trials": n_trials,
                "baseline_attack": str(baseline_attack),
            },
        )

        return heatmap_data, selected_trials_info

    def select_trial_for_model_attack(
        self,
        study_data: StudyData,
        attack: AttackName,
        baseline_attack: AttackName,
        baseline_utility: float | None,
        epsilon: float,
        n_trials: int,
        epsilon_filter: EpsilonBoundedFilter,
    ) -> optuna.trial.FrozenTrial | None:
        """Dispatch to appropriate trial selection method based on attack type."""
        if attack == baseline_attack:
            return self._select_trial_for_baseline_attack(study_data)

        if attack == AttackName.EMBEDDING_ATTACK:
            return self._select_trial_for_embedding_attack(study_data)

        return self._select_trial_with_epsilon_filter(
            study_data=study_data,
            baseline_utility=baseline_utility,
            epsilon=epsilon,
            n_trials=n_trials,
            epsilon_filter=epsilon_filter,
        )

    def extract_metrics_for_heatmap(
        self,
        selected_trial: optuna.trial.FrozenTrial,
        attack: AttackName,
        model: str,
        all_data: dict[str, dict[AttackName, StudyData]],
        baseline_attack: AttackName,
        harmfulness_metric_name: str,
        utility_metric_name: str,
    ) -> tuple[float | None, float | None]:
        """Extract (harmfulness, utility) metrics for heatmap cell, handling embedding_attack specially."""
        # Embedding attack special case: uses embedding_attack_eval for harmfulness, same utility as baseline
        if attack == AttackName.EMBEDDING_ATTACK:
            harmfulness_val = get_metric_value(selected_trial, str(EvalName.EMBEDDING_ATTACK), self.config)
            utility_val = self._get_embedding_attack_utility(
                model=model,
                all_data=all_data,
                baseline_attack=baseline_attack,
                utility_metric_name=utility_metric_name,
            )
            return harmfulness_val, utility_val

        harmfulness_val = get_metric_value(selected_trial, harmfulness_metric_name, self.config)
        utility_val = get_metric_value(selected_trial, utility_metric_name, self.config)

        return harmfulness_val, utility_val

    def copy_trial_artifacts(
        self,
        selected_trials_info: dict[tuple[str, AttackName], dict[str, Any]],
        all_data: dict[str, dict[AttackName, StudyData]],
        output_dir: Path,
    ) -> None:
        """Copy trial configurations and evaluation results to output directory."""
        LOGGER.info(
            f"Copying best trial configs and evaluations for {len(selected_trials_info)} model/attack combinations..."
        )

        for (model, attack), trial_info in selected_trials_info.items():
            trial_dir: Path = trial_info["trial_dir"]
            trial_number: int = trial_info["trial_number"]

            model_attack_dir = output_dir / model / str(attack)
            ensure_dir(model_attack_dir)

            # Write trial parameters to grid.yaml
            if model in all_data and attack in all_data[model]:
                study_data = all_data[model][attack]
                trial = study_data.study.trials[trial_number]

                grid_yaml_path = model_attack_dir / ConfigPath.GRID_YAML
                grid_data: dict[str, dict[str, object]] = {
                    f"{model}_params": dict(trial.user_attrs[OptunaUserAttrs.MERGED_CONFIG]),
                }
                with open(grid_yaml_path, "w") as f:
                    yaml.dump(grid_data, f, default_flow_style=False, sort_keys=False)
                LOGGER.debug(f"Wrote grid params for {model}/{attack}/{get_trial_dir_name(trial_number)}")

            # Copy trial config if it exists
            trial_config_file = trial_dir / OptunaPaths.TRIAL_CONFIG
            if trial_config_file.exists():
                dest_config = model_attack_dir / f"{get_trial_dir_name(trial_number)}_config.yaml"
                shutil.copy2(trial_config_file, dest_config)
                LOGGER.debug(f"Copied config for {model}/{attack}/{get_trial_dir_name(trial_number)}")

            # Copy evaluation directory if it exists
            eval_dir = trial_dir / EvalPath.CHECKPOINT_EVAL_DIR
            if eval_dir.exists() and eval_dir.is_dir():
                dest_eval_dir = model_attack_dir / get_trial_eval_dir_name(trial_number)
                if dest_eval_dir.exists():
                    shutil.rmtree(dest_eval_dir)
                shutil.copytree(eval_dir, dest_eval_dir)
                LOGGER.debug(f"Copied evaluation dir for {model}/{attack}/{get_trial_dir_name(trial_number)}")

    def _select_trial_for_baseline_attack(
        self,
        study_data: StudyData,
    ) -> optuna.trial.FrozenTrial | None:
        """Return first completed trial for baseline attack (no filtering)."""
        completed_trials = [t for t in study_data.study.get_trials() if t.state.value == 1]
        return completed_trials[0] if completed_trials else None

    def _select_trial_for_embedding_attack(
        self,
        study_data: StudyData,
    ) -> optuna.trial.FrozenTrial | None:
        """Return trial with maximum embedding_attack_eval metric (no epsilon filtering)."""
        embedding_attack_eval_key = str(EvalName.EMBEDDING_ATTACK)
        completed_trials = [
            t
            for t in study_data.study.get_trials()
            if t.state.value == 1  # Complete trials
            and OptunaUserAttrs.EVAL_METRICS in t.user_attrs
            and isinstance(t.user_attrs[OptunaUserAttrs.EVAL_METRICS], dict)
            and embedding_attack_eval_key in t.user_attrs[OptunaUserAttrs.EVAL_METRICS]
        ]

        if not completed_trials:
            return None

        return max(
            completed_trials,
            key=lambda t: t.user_attrs[OptunaUserAttrs.EVAL_METRICS][embedding_attack_eval_key],
        )

    def _select_trial_with_epsilon_filter(
        self,
        study_data: StudyData,
        baseline_utility: float | None,
        epsilon: float,
        n_trials: int,
        epsilon_filter: EpsilonBoundedFilter,
    ) -> optuna.trial.FrozenTrial | None:
        """Filter trials by utility constraint and return best by safety metric."""
        if baseline_utility is None:
            LOGGER.warning("No baseline utility available, skipping epsilon filtering")
            return None

        filtered_trials = epsilon_filter.filter_trials(
            study_data.study,
            baseline_utility,
            epsilon,
            n_trials=n_trials,
            analysis_config=self.config,
        )

        if not filtered_trials:
            LOGGER.debug("No trials passed epsilon bound")
            return None

        return epsilon_filter.select_best_trial(
            filtered_trials,
            analysis_config=self.config,
        )

    def _get_embedding_attack_utility(
        self,
        model: str,
        all_data: dict[str, dict[AttackName, StudyData]],
        baseline_attack: AttackName,
        utility_metric_name: str,
    ) -> float | None:
        """Return baseline utility score for embedding_attack (no weight modification)."""
        if baseline_attack not in all_data.get(model, {}):
            return None

        baseline_study_data = all_data[model][baseline_attack]
        baseline_trials = [
            t
            for t in baseline_study_data.study.get_trials()
            if t.state.value == 1  # COMPLETE
        ]

        if not baseline_trials:
            return None

        baseline_trial = baseline_trials[0]
        return get_metric_value(baseline_trial, utility_metric_name, self.config)
