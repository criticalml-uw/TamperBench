"""GaLore fine-tuning attack."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack
from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetuneConfig,
)
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.names import AttackName


@dataclass
class GaloreFinetuneConfig(FullParameterFinetuneConfig):
    """Hyperparameters for a GaLore fine-tuning attack.

    Attributes:
        galore_rank: Rank of the low-rank gradient projection.
        galore_update_proj_gap: Number of optimizer steps between projection updates.
        galore_scale: Scale applied to projected gradients.
        galore_proj_type: Projection orientation used by the GaLore optimizer.
        optim_target_modules: Linear modules whose gradients are projected by GaLore.
    """

    galore_rank: int = field(default=128, kw_only=True)
    galore_update_proj_gap: int = field(default=200, kw_only=True)
    galore_scale: float = field(default=0.25, kw_only=True)
    galore_proj_type: str = field(default="std", kw_only=True)
    optim_target_modules: str | list[str] | None = field(default=None, kw_only=True)


@register_attack(AttackName.GALORE_FINETUNE, GaloreFinetuneConfig)
class GaloreFinetune(TamperAttack[GaloreFinetuneConfig]):
    """Galore Finetune attack implementation."""

    name: AttackName = AttackName.GALORE_FINETUNE

    @override
    def run_attack(self) -> None:
        """Execute the attack.

        Must save checkpoint to self.output_checkpoint_path.
        """
        raise NotImplementedError("GaLore fine-tuning is not Implemented yet")
