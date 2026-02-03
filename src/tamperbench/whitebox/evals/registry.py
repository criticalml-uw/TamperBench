"""Evaluation registry with decorator-based registration.

Allows individual evaluation modules to self-register their mapping from
`EvalName -> EvaluationClass` at import time, avoiding a central hard-coded map.
"""

# pyright: reportMissingTypeArgument=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation
from tamperbench.whitebox.utils.names import EvalName

E = TypeVar("E", bound=WhiteBoxEvaluation)


EVALS_REGISTRY: dict[EvalName, type[WhiteBoxEvaluation]] = {}


def register_evaluation(name: EvalName) -> Callable[[type[E]], type[E]]:
    """Register an evaluation class under a name."""

    def _decorator(eval_cls: type[E]) -> type[E]:
        EVALS_REGISTRY[name] = eval_cls
        return eval_cls

    return _decorator
