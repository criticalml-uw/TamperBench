"""Evaluation registry with decorator-based registration.

Allows individual evaluation modules to self-register their mapping from
`EvalName -> EvaluationClass` at import time, avoiding a central hard-coded map.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from safetunebed.whitebox.evals.base import WhiteBoxEvaluation
from safetunebed.whitebox.utils.names import EvalName

E = TypeVar("E", bound=WhiteBoxEvaluation)  # pyright: ignore[reportMissingTypeArgument]


EVALS_REGISTRY: dict[EvalName, type[WhiteBoxEvaluation]] = {}  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]


def register_evaluation(name: EvalName) -> Callable[[type[E]], type[E]]:
    """Decorator to register an evaluation class under a name."""  # noqa: D401

    def _decorator(eval_cls: type[E]) -> type[E]:
        EVALS_REGISTRY[name] = eval_cls
        return eval_cls

    return _decorator
