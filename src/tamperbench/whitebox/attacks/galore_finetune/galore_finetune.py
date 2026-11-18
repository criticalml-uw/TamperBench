"""GaLore fine-tuning attack."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from trl.trainer.sft_config import SFTConfig
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack
from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
    run_full_finetune_attack,
)
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.names import AttackName
from tamperbench.whitebox.utils.ops import run_in_isolation


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
        # Prepare data
        dataset_loader = FullParameterFinetune(attack_config=self.attack_config)
        prompt_completions_dataset = dataset_loader.load_prompt_completions_dataset()

        # Setup training
        training_arguments = SFTConfig(
            output_dir=self.attack_config.out_dir,
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            learning_rate=self.attack_config.learning_rate,
            num_train_epochs=self.attack_config.num_train_epochs,
            max_steps=self.attack_config.max_steps,
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            completion_only_loss=True,
            max_length=2048,
            seed=self.attack_config.random_seed,
            use_cpu=not torch.cuda.is_available(),
            report_to="none",
            optim=self.attack_config.optim,
            optim_args=(
                f"rank={self.attack_config.galore_rank},"
                f"update_proj_gap={self.attack_config.galore_update_proj_gap},"
                f"scale={self.attack_config.galore_scale},"
                f"proj_type={self.attack_config.galore_proj_type}"
            ),
            optim_target_modules=self.attack_config.optim_target_modules,
            warmup_steps=self.attack_config.warmup_steps,
            warmup_ratio=self.attack_config.warmup_ratio,
            gradient_accumulation_steps=self.attack_config.gradient_accumulation_steps,
        )

        # Do training attack
        run_in_isolation(
            target=run_full_finetune_attack,
            args=(
                self.attack_config,
                training_arguments,
                prompt_completions_dataset,
                self.output_checkpoint_path,
            ),
            error_context="GaLore finetune attack",
        )
