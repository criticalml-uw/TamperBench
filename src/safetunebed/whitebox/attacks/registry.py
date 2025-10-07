"""Attack registry with decorator-based registration.

Allows individual attack modules to self-register their mapping from
`AttackName -> (ConfigDataclass, AttackClass)` at import time, avoiding a
central hard-coded map.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.utils.names import AttackName

H = TypeVar("H", bound=TamperAttackConfig)
T = TypeVar("T", bound=TamperAttack)  # pyright: ignore[reportMissingTypeArgument]


ATTACKS_REGISTRY: dict[  # pyright: ignore[reportUnknownVariableType]
    AttackName, tuple[type[TamperAttackConfig], type[TamperAttack]]  # pyright: ignore[reportMissingTypeArgument]
] = {}


def register_attack(
    name: AttackName, config_cls: type[H]
) -> Callable[[type[T]], type[T]]:
    """Decorator to register an attack class and its config class under a name."""  # noqa: D401

    def _decorator(attack_cls: type[T]) -> type[T]:
        ATTACKS_REGISTRY[name] = (config_cls, attack_cls)
        return attack_cls

    return _decorator
