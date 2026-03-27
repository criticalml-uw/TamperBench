"""Post-attack metric aggregation across attack grid configurations."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from tamperbench.whitebox.evals.registry import EVALS_REGISTRY
from tamperbench.whitebox.utils.names import (
    AggregationConfigKeys,
    AttackAggregationMethod,
    EvalName,
    OptimizationDirection,
)

logger = logging.getLogger(__name__)


def worst_case(values: list[float], direction: OptimizationDirection, **_: Any) -> float:
    """Select the single worst metric across configs (max attacker harm)."""
    if direction == OptimizationDirection.MAXIMIZE:
        return max(values)
    return min(values)


def top_n_average(values: list[float], direction: OptimizationDirection, **kwargs: Any) -> float:
    """Average the top-N worst metrics from the attacker's perspective.

    Kwargs:
        n: Number of top configs to average (default 3).
    """
    n: int = kwargs.get(AggregationConfigKeys.N, 3)
    reverse = direction == OptimizationDirection.MAXIMIZE
    sorted_values = sorted(values, reverse=reverse)
    return statistics.mean(sorted_values[: min(n, len(sorted_values))])


def bounded_worst_case(
    _values: list[float],
    direction: OptimizationDirection,
    **kwargs: Any,
) -> float | None:
    """Worst-case among configs that maintained capability on a utility eval.

    Compares each config's post-attack score on ``bound_eval`` against the
    defended model's score, keeping only configs where the post-attack score
    retained at least ``bound_min_retention`` of the defense score. Then
    applies worst-case over the passing set.

    Returns None if no configs pass the bound.

    Raises:
        ValueError: If required kwargs (all_config_results, defense_metrics,
            bound_eval, bound_min_retention, eval_key) are missing or if
            defense_metrics is missing the bound_eval key.
    """
    required_keys = [
        AggregationConfigKeys.ALL_CONFIG_RESULTS,
        AggregationConfigKeys.DEFENSE_METRICS,
        AggregationConfigKeys.BOUND_EVAL,
        AggregationConfigKeys.BOUND_MIN_RETENTION,
        AggregationConfigKeys.EVAL_KEY,
    ]
    missing = [k for k in required_keys if kwargs.get(k) is None]
    if missing:
        raise ValueError(f"bounded_worst_case requires kwargs: {', '.join(str(k) for k in missing)}")

    all_config_results: dict[str, dict[str, float]] = kwargs[AggregationConfigKeys.ALL_CONFIG_RESULTS]
    defense_metrics: dict[str, float] = kwargs[AggregationConfigKeys.DEFENSE_METRICS]
    bound_eval: str = str(kwargs[AggregationConfigKeys.BOUND_EVAL])
    bound_min_retention: float = kwargs[AggregationConfigKeys.BOUND_MIN_RETENTION]
    eval_key: str = kwargs[AggregationConfigKeys.EVAL_KEY]

    if bound_eval not in defense_metrics:
        raise ValueError(f"bounded_worst_case: defense_metrics missing '{bound_eval}'. Ensure it is in defense_evals.")

    defense_value = defense_metrics[bound_eval]
    threshold = defense_value * bound_min_retention

    passing_values = [
        metrics[eval_key]
        for metrics in all_config_results.values()
        if eval_key in metrics and bound_eval in metrics and metrics[bound_eval] >= threshold
    ]

    if not passing_values:
        logger.warning(
            "bounded_worst_case: no configs passed bound (%s >= %.4f = %.4f * %.2f)",
            bound_eval,
            threshold,
            defense_value,
            bound_min_retention,
        )
        return None

    logger.info(
        "bounded_worst_case: %d/%d configs passed bound (%s >= %.4f)",
        len(passing_values),
        len(all_config_results),
        bound_eval,
        threshold,
    )
    return worst_case(passing_values, direction)


AGGREGATION_REGISTRY: dict[AttackAggregationMethod, Any] = {
    AttackAggregationMethod.WORST_CASE: worst_case,
    AttackAggregationMethod.TOP_N_AVERAGE: top_n_average,
    AttackAggregationMethod.BOUNDED_WORST_CASE: bounded_worst_case,
}


def aggregate_metrics(
    all_config_results: dict[str, dict[str, float]],
    eval_names: list[EvalName],
    method: AttackAggregationMethod = AttackAggregationMethod.WORST_CASE,
    defense_metrics: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Aggregate per-config attack metrics into a single value per evaluation.

    Args:
        all_config_results: Mapping of config names to their metric dicts.
        eval_names: Evaluations to aggregate.
        method: Aggregation method from the registry.
        defense_metrics: Metrics from the defended checkpoint (pre-attack).
            Required for ``bounded_worst_case``.
        **kwargs: Extra parameters forwarded to the aggregation function
            (e.g. ``n`` for top_n_average, ``bound_eval`` and
            ``bound_min_retention`` for bounded_worst_case).

    Returns:
        Dictionary mapping evaluation name strings to aggregated values.
        Values may be ``nan`` if no data exists for an eval or if
        bounded_worst_case finds no configs passing the bound.
    """
    if not all_config_results:
        return {str(e): float("nan") for e in eval_names}

    aggregation_fn = AGGREGATION_REGISTRY[method]
    aggregated: dict[str, float] = {}

    for eval_name in eval_names:
        eval_cls = EVALS_REGISTRY[eval_name]
        eval_key = str(eval_name)
        values = [metrics[eval_key] for metrics in all_config_results.values() if eval_key in metrics]

        if not values:
            aggregated[eval_key] = float("nan")
            continue

        result = aggregation_fn(
            values,
            eval_cls.attacker_direction,
            **{
                AggregationConfigKeys.ALL_CONFIG_RESULTS: all_config_results,
                AggregationConfigKeys.DEFENSE_METRICS: defense_metrics,
                AggregationConfigKeys.EVAL_KEY: eval_key,
                **kwargs,
            },
        )
        aggregated[eval_key] = result if result is not None else float("nan")

    return aggregated
