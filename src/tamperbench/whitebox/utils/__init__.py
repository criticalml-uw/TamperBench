"""Defined templates for attacks, defenses and evals in tamperbench to define form factors."""

from tamperbench.whitebox.utils.datasets import get_dataset
from tamperbench.whitebox.utils.generation import (
    REFUSAL_INDICATORS,
    clear_memory,
    count_refusals,
    generate_responses,
    is_refusal,
)
from tamperbench.whitebox.utils.names import (
    AttackName,
    ConfigPath,
    EvalName,
    EvalPath,
    MetricName,
    OptimizationDirection,
)
from tamperbench.whitebox.utils.ops import dealloc_model_and_tokenizer

__all__ = [
    "REFUSAL_INDICATORS",
    "AttackName",
    "ConfigPath",
    "EvalName",
    "EvalPath",
    "MetricName",
    "OptimizationDirection",
    "clear_memory",
    "count_refusals",
    "dealloc_model_and_tokenizer",
    "generate_responses",
    "get_dataset",
    "is_refusal",
]
