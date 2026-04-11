"""Trial filtering utilities for epsilon-bounded analysis.

Provides EpsilonBoundedFilter for selecting trials that satisfy utility degradation
constraints while maximizing safety metrics.
"""

from __future__ import annotations

import logging

import optuna

from tamperbench.whitebox.utils.analysis.config import (
    AnalysisConfig,
    EpsilonBoundedConfig,
)
from tamperbench.whitebox.utils.analysis.metrics import (
    calculate_utility_drop,
    get_metric_value,
)
from tamperbench.whitebox.utils.names import MetricName

LOGGER = logging.getLogger(__name__)


class EpsilonBoundedFilter:
    """Filter trials with epsilon-bounded MMLU drop."""

    config: EpsilonBoundedConfig

    def __init__(self, config: EpsilonBoundedConfig):
        """Initialize filter with configuration.

        Args:
            config: Epsilon-bounded configuration
        """
        self.config = config

    def filter_trials(
        self,
        study: optuna.Study,
        baseline_utility: float,
        epsilon: float,
        n_trials: int | None = None,
        analysis_config: AnalysisConfig | None = None,
    ) -> list[optuna.trial.FrozenTrial]:
        """Filter trials by epsilon-bounded utility degradation.

        Selects trials where utility degradation from baseline is within epsilon threshold.

        Args:
            study: Optuna study containing trials
            baseline_utility: Baseline utility score for comparison
            epsilon: Maximum allowed utility drop as fraction (e.g., 0.05 for 5%)
            n_trials: Maximum number of chronological trials to consider
            analysis_config: Configuration for metric extraction

        Returns:
            Filtered trials satisfying epsilon constraint

        Example:
            >>> filter = EpsilonBoundedFilter(config.epsilon_bounded)
            >>> filtered = filter.filter_trials(study, baseline_utility=0.75, epsilon=0.05)
            >>> len(filtered)  # Trials with <= 5% utility drop
            12
        """
        trials_to_filter = self.get_completed_trials(study, n_trials)
        filtered: list[optuna.trial.FrozenTrial] = []

        for trial in trials_to_filter:
            utility_value = self.extract_utility_metric(trial, analysis_config)

            if utility_value is None:
                self.log_missing_utility(trial, analysis_config)
                continue

            if self.satisfies_epsilon_constraint(utility_value, baseline_utility, epsilon, trial):
                filtered.append(trial)

        self.log_filter_results(len(filtered), len(trials_to_filter), epsilon)
        return filtered

    def select_best_trial(
        self,
        filtered_trials: list[optuna.trial.FrozenTrial],
        analysis_config: AnalysisConfig | None = None,
    ) -> optuna.trial.FrozenTrial | None:
        """Select trial with maximum safety metric from filtered trials.

        Args:
            filtered_trials: Pre-filtered trials
            analysis_config: Configuration for metric extraction

        Returns:
            Trial with highest safety metric, or None if list empty

        Example:
            >>> filtered = filter.filter_trials(study, baseline_utility=0.75, epsilon=0.05)
            >>> best = filter.select_best_trial(filtered, analysis_config=config)
            >>> best.number
            42
        """
        if not filtered_trials:
            return None

        best_trial = None
        best_safety = float("-inf")

        for trial in filtered_trials:
            safety_value = self.extract_safety_metric(trial, analysis_config)

            if safety_value is not None and safety_value > best_safety:
                best_safety = safety_value
                best_trial = trial

        if best_trial:
            LOGGER.info(f"Selected trial {best_trial.number} with safety={best_safety:.3f}")

        return best_trial

    def get_completed_trials(
        self,
        study: optuna.Study,
        n_trials: int | None = None,
    ) -> list[optuna.trial.FrozenTrial]:
        """Get completed trials sorted by trial number, optionally limited to n_trials.

        Args:
            study: Optuna study to get trials from
            n_trials: Maximum number of trials to return (None for all)

        Returns:
            List of completed trials sorted by trial number
        """
        completed = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.number)

        if n_trials is not None:
            completed = completed[:n_trials]

        return completed

    def extract_utility_metric(
        self,
        trial: optuna.trial.FrozenTrial,
        analysis_config: AnalysisConfig | None,
    ) -> float | None:
        """Extract utility metric value from trial.

        Args:
            trial: Optuna frozen trial
            analysis_config: Configuration for metric extraction (uses trial.values if None)

        Returns:
            Utility metric value, or None if not found
        """
        if analysis_config:
            utility_metric_key = analysis_config.epsilon_bounded.utility_metric
            return get_metric_value(trial, utility_metric_key, analysis_config)

        values = trial.values
        return values[1] if values and len(values) >= 2 else None

    def extract_safety_metric(
        self,
        trial: optuna.trial.FrozenTrial,
        analysis_config: AnalysisConfig | None,
    ) -> float | None:
        """Extract safety metric value from trial.

        Args:
            trial: Optuna frozen trial
            analysis_config: Configuration for metric extraction (uses trial.values if None)

        Returns:
            Safety metric value, or None if not found
        """
        if analysis_config:
            safety_metric_key = MetricName(analysis_config.epsilon_bounded.safety_metric)
            return get_metric_value(trial, safety_metric_key, analysis_config)

        values = trial.values
        return values[0] if values and len(values) >= 1 else None

    def satisfies_epsilon_constraint(
        self,
        utility_value: float,
        baseline_utility: float,
        epsilon: float,
        trial: optuna.trial.FrozenTrial,
    ) -> bool:
        """Check if utility drop from baseline is within epsilon threshold.

        Args:
            utility_value: Trial's utility score
            baseline_utility: Baseline utility score
            epsilon: Maximum allowed utility drop fraction
            trial: Trial being checked (for logging)

        Returns:
            True if utility drop is within epsilon threshold
        """
        utility_drop = calculate_utility_drop(utility_value, baseline_utility)

        if utility_drop <= epsilon:
            return True

        LOGGER.debug(f"Trial {trial.number}: utility drop {utility_drop:.3f} exceeds epsilon {epsilon}")
        return False

    def log_missing_utility(
        self,
        trial: optuna.trial.FrozenTrial,
        analysis_config: AnalysisConfig | None,
    ) -> None:
        """Log warning when utility metric cannot be extracted from trial.

        Args:
            trial: Trial that failed metric extraction
            analysis_config: Configuration (used to get metric name for logging)
        """
        util_key = analysis_config.epsilon_bounded.utility_metric if analysis_config else "utility"
        LOGGER.warning(f"Trial {trial.number}: Could not extract {util_key}, skipping")

    def log_filter_results(
        self,
        filtered_count: int,
        total_count: int,
        epsilon: float,
    ) -> None:
        """Log summary of how many trials passed the epsilon filter.

        Args:
            filtered_count: Number of trials that passed filtering
            total_count: Total number of trials considered
            epsilon: Epsilon threshold used for filtering
        """
        LOGGER.info(f"Filtered to {filtered_count}/{total_count} trials with utility drop ≤ {epsilon:.1%}")
