"""GCG attack interface."""

from dataclasses import dataclass, field

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.gcg.gcg import GCGEvaluation, GCGEvaluationConfig
from tamperbench.whitebox.evals.gcg.implementation import GCGConfig
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.names import AttackName, EvalName


@dataclass
class GCGAttackConfig(TamperAttackConfig):
    """Config for GCG attack."""

    gcg_config: GCGConfig = field(default_factory=GCGConfig)


@register_attack(AttackName.GCG_ATTACK, GCGAttackConfig)
class GCGAttack(TamperAttack[GCGAttackConfig]):
    """GCG attack class."""

    name: AttackName = AttackName.GCG_ATTACK

    @override
    def run_attack(self) -> None:
        """GCG attack does not modify model weights directly."""
        self.output_checkpoint_path: str = self.attack_config.input_checkpoint_path

    @override
    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate GCG attack, including GCG-specific evaluation if requested."""
        results = super().evaluate()
        if EvalName.GCG_ATTACK_EVAL in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_gcg()])
        return EvaluationSchema.validate(results)

    def _evaluate_gcg(self) -> DataFrame[EvaluationSchema]:
        """Instantiate GCGEvaluation and run evaluation."""
        eval_cfg = GCGEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            gcg_config=self.attack_config.gcg_config,
        )
        evaluator = GCGEvaluation(eval_cfg)
        return evaluator.run_evaluation()
