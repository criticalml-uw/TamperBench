"""Defense registry with decorator-based registration."""

# pyright: reportMissingTypeArgument=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.utils.names import DefenseName

T = TypeVar("T", bound=AlignmentDefense)


DEFENSES_REGISTRY: dict[DefenseName, tuple[type[AlignmentDefenseConfig], type[AlignmentDefense]]] = {}


def register_defense(name: DefenseName, config_cls: type[AlignmentDefenseConfig]) -> Callable[[type[T]], type[T]]:
    """Register a defense class and its config class under a name."""

    def _decorator(defense_cls: type[T]) -> type[T]:
        DEFENSES_REGISTRY[name] = (config_cls, defense_cls)
        return defense_cls

    return _decorator
