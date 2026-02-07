"""Latent attack wrapper."""

from dataclasses import dataclass

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.latent_attack import (
    LatentAttackEvaluation,
    LatentAttackEvaluationConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.names import AttackName, EvalName


@dataclass
class LatentAttackConfig(TamperAttackConfig):
    """Config for latent attack wrapper.

    Attributes:
        epsilon: L-infinity norm bound for the latent perturbation.
        attack_layer: Dot-path to module to perturb (e.g., "transformer.h.0.mlp").
        small: Whether to use the small dataset for testing.
    """

    epsilon: float
    attack_layer: str
    small: bool


@register_attack(AttackName.LATENT_ATTACK, LatentAttackConfig)
class LatentAttack(TamperAttack[LatentAttackConfig]):
    """Latent attack class."""

    name: AttackName = AttackName.LATENT_ATTACK

    @override
    def run_attack(self) -> None:
        """Latent attack does not modify weights directly."""

    @override
    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate latent attack by running the evaluator."""
        results = super().evaluate()
        if EvalName.LATENT_ATTACK in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_latent_attack()])
        return EvaluationSchema.validate(results)

    def _evaluate_latent_attack(self) -> DataFrame[EvaluationSchema]:
        """Instantiate LatentAttackEvaluation and run evaluation."""
        eval_cfg = LatentAttackEvaluationConfig(
            model_checkpoint=self.attack_config.input_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            epsilon=self.attack_config.epsilon,
            attack_layer=self.attack_config.attack_layer,
        )
        evaluator = LatentAttackEvaluation(eval_cfg)
        return evaluator.run_evaluation()
