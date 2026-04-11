"""Evaluation registry with decorator-based registration.

Allows individual evaluation modules to self-register their mapping from
`EvalName -> EvaluationClass` at import time, avoiding a central hard-coded map.
"""

# pyright: reportMissingTypeArgument=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from tamperbench.whitebox.utils.names import EvalName

E = TypeVar("E", bound=WhiteBoxEvaluation)


EVALS_REGISTRY: dict[EvalName, type[WhiteBoxEvaluation]] = {}
EVAL_CONFIG_REGISTRY: dict[EvalName, type[WhiteBoxEvaluationConfig]] = {}


def register_evaluation(
    name: EvalName,
    config_cls: type[WhiteBoxEvaluationConfig] = WhiteBoxEvaluationConfig,
) -> Callable[[type[E]], type[E]]:
    """Register an evaluation class and its config class under a name."""

    def _decorator(eval_cls: type[E]) -> type[E]:
        EVALS_REGISTRY[name] = eval_cls
        EVAL_CONFIG_REGISTRY[name] = config_cls
        return eval_cls

    return _decorator
