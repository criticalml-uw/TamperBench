"""Metric extraction and normalization utilities.

Provides functions for extracting metrics from Optuna trials, normalizing metric
names using aliases, and calculating metric drop percentages.
"""

from __future__ import annotations

import logging

import optuna

from tamperbench.whitebox.utils.analysis.config import AnalysisConfig
from tamperbench.whitebox.utils.names import EvalName, MetricName, OptunaUserAttrs

LOGGER = logging.getLogger(__name__)


def resolve_eval_to_metric_key(eval_name: str | EvalName) -> str:
    """Map eval name to its primary metric key.

    Some evaluations report metrics under different keys than their eval name.
    This function handles the mapping (e.g., embedding_attack -> strong_reject).

    Args:
        eval_name: Evaluation name as string or EvalName enum

    Returns:
        The metric key to use for looking up this evaluation's results

    Example:
        >>> resolve_eval_to_metric_key(EvalName.EMBEDDING_ATTACK)
        'strong_reject'
        >>> resolve_eval_to_metric_key("mmlu_pro_val")
        'mmlu_pro_val'
    """
    eval_name_str = str(eval_name)

    if eval_name_str == str(EvalName.EMBEDDING_ATTACK):
        return MetricName.STRONG_REJECT

    return eval_name_str


def extract_trial_metrics(
    trial: optuna.trial.FrozenTrial,
    config: AnalysisConfig,
) -> dict[str, float]:
    """Extract all metrics from a trial, preferring user_attrs over trial.values.

    Looks for metrics in trial.user_attrs['eval_metrics'] first. If not found,
    falls back to extracting from trial.values by position index.

    Args:
        trial: Optuna frozen trial containing metric data
        config: Analysis configuration with metric definitions

    Returns:
        Dictionary mapping metric names to their float values

    Example:
        >>> metrics = extract_trial_metrics(trial, config)
        >>> metrics.get("strong_reject")
        0.85
    """
    metrics = _extract_from_user_attrs(trial)
    if metrics:
        return metrics

    return _extract_from_trial_values(trial, config)


def get_metric_value(
    trial: optuna.trial.FrozenTrial,
    metric_name: str,
    config: AnalysisConfig,
) -> float | None:
    """Get a specific metric value from trial by exact name or alias lookup.

    First attempts exact match on metric_name, then searches through configured
    metric aliases to find a match.

    Args:
        trial: Optuna frozen trial containing metric data
        metric_name: Name of metric to retrieve (can be an alias)
        config: Analysis configuration with metric alias definitions

    Returns:
        Metric value as float, or None if not found

    Example:
        >>> get_metric_value(trial, "strong_reject", config)
        0.85
        >>> get_metric_value(trial, "mmlu_pro_val", config)
        0.72
    """
    metrics = extract_trial_metrics(trial, config)

    value = _find_by_exact_match(metrics, metric_name)
    if value is not None:
        return value

    return _find_by_alias_match(metrics, metric_name, config)


def calculate_utility_drop(
    trial_utility: float,
    baseline_utility: float,
) -> float:
    """Calculate utility drop as fraction: (baseline - trial) / baseline.

    Computes how much utility has degraded relative to baseline. Result is
    clamped to [0, 1] range (negative drops, i.e. improvements, return 0).

    Args:
        trial_utility: Utility score from the trial being evaluated
        baseline_utility: Reference utility score (typically from unmodified model)

    Returns:
        Fractional drop in utility, clamped to [0, 1]

    Example:
        >>> calculate_utility_drop(0.70, 0.75)
        0.0667  # 6.67% drop
        >>> calculate_utility_drop(0.80, 0.75)
        0.0  # No drop (improvement clamped to 0)
    """
    if baseline_utility == 0:
        return 0.0

    drop = (baseline_utility - trial_utility) / baseline_utility
    return max(0.0, drop)


def format_metric_value(value: float | None, precision: int = 3) -> str:
    """Format metric value for display, returning 'N/A' if None.

    Args:
        value: Metric value to format, or None
        precision: Number of decimal places (default: 3)

    Returns:
        Formatted string representation of the value

    Example:
        >>> format_metric_value(0.85432)
        '0.854'
        >>> format_metric_value(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _extract_from_user_attrs(trial: optuna.trial.FrozenTrial) -> dict[str, float]:
    """Extract metrics dict from trial.user_attrs['eval_metrics'], or empty dict."""
    eval_metrics = trial.user_attrs.get(OptunaUserAttrs.EVAL_METRICS)
    if not isinstance(eval_metrics, dict):
        return {}

    metrics: dict[str, float] = {}
    for key, value in eval_metrics.items():  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, int | float):
            metrics[str(key)] = float(value)

    return metrics


def _extract_from_trial_values(trial: optuna.trial.FrozenTrial, config: AnalysisConfig) -> dict[str, float]:
    """Extract metrics from trial.values by index (0=SR, 1=MMLU)."""
    metrics: dict[str, float] = {}

    values = trial.values  # pyright: ignore[reportAny]
    if values is None:
        return metrics

    if len(values) >= 1:  # pyright: ignore[reportAny]
        sr_metric = config.metrics.get("strong_reject")
        if sr_metric:
            metrics[sr_metric.aliases[0]] = float(values[0])  # pyright: ignore[reportAny]

    if len(values) >= 2:  # pyright: ignore[reportAny]
        mmlu_metric = config.metrics.get("mmlu_pro")
        if mmlu_metric:
            metrics[mmlu_metric.aliases[1]] = float(values[1])  # pyright: ignore[reportAny]

    return metrics


def _find_by_exact_match(metrics: dict[str, float], metric_name: str) -> float | None:
    """Return metric value if exact name exists in dict, else None."""
    return metrics.get(metric_name)


def _find_by_alias_match(metrics: dict[str, float], metric_name: str, config: AnalysisConfig) -> float | None:
    """Search config aliases for metric_name and return value if any alias matches."""
    metric_config = config.get_metric_by_alias(metric_name)
    if not metric_config:
        return None

    for alias in metric_config.aliases:
        if alias in metrics:
            return metrics[alias]

    return None
