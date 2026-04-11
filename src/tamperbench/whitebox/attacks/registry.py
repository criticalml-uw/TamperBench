"""Attack registry with decorator-based registration.

Allows individual attack modules to self-register their mapping from
`AttackName -> (ConfigDataclass, AttackClass)` at import time, avoiding a
central hard-coded map.
"""

# pyright: reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.utils.names import AttackName

T = TypeVar("T", bound=TamperAttack)


ATTACKS_REGISTRY: dict[AttackName, tuple[type[TamperAttackConfig], type[TamperAttack]]] = {}


def register_attack(name: AttackName, config_cls: type[TamperAttackConfig]) -> Callable[[type[T]], type[T]]:
    """Register an attack class and its config class under a name."""

    def _decorator(attack_cls: type[T]) -> type[T]:
        ATTACKS_REGISTRY[name] = (config_cls, attack_cls)
        return attack_cls

    return _decorator


@register_attack(AttackName.NO_WEIGHT_MODIFICATION, TamperAttackConfig)
class NoWeightModificationAttack(TamperAttack[TamperAttackConfig]):
    """No tamper attack (generally used purely for evaluations - or for non-weight modifying eval)."""

    name: AttackName = AttackName.NO_WEIGHT_MODIFICATION

    @override
    def run_attack(self) -> None:
        """No tamper attack takes place so the output is the input."""
        self.output_checkpoint_path = self.attack_config.input_checkpoint_path
        pass
