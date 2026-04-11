"""Prompt attack template wrapper.

NOTE: This is conceptually a blackbox attack (it only manipulates input text, not model weights).
It lives in the whitebox package to reuse the existing evaluation pipeline and serve as a baseline
for comparison against weight-modification attacks.
"""

from dataclasses import dataclass

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.prompt_attack import (
    PromptAttackEvaluation,
    PromptAttackEvaluationConfig,
)
from tamperbench.whitebox.utils.names import AttackName, EvalName


@dataclass
class PromptAttackConfig(TamperAttackConfig):
    """Config for prompt jailbreak attack wrapper."""

    jailbreak_prompt: str = "aim"


@register_attack(AttackName.PROMPT_ATTACK, PromptAttackConfig)
class PromptAttack(TamperAttack[PromptAttackConfig]):
    """Prompt attack class."""

    name: AttackName = AttackName.PROMPT_ATTACK

    @override
    def run_attack(self) -> None:
        """Prompt attack does not modify model weights directly."""
        self.output_checkpoint_path: str = self.attack_config.input_checkpoint_path

    @override
    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate prompt attack by running the evaluator."""
        results = super().evaluate()
        if EvalName.PROMPT_ATTACK in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_prompt_attack()])
        return EvaluationSchema.validate(results)

    def _evaluate_prompt_attack(self) -> DataFrame[EvaluationSchema]:
        """Instantiate PromptAttackEvaluation and run evaluation."""
        eval_cfg = PromptAttackEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            jailbreak_prompt=self.attack_config.jailbreak_prompt,
        )
        evaluator = PromptAttackEvaluation(eval_cfg)
        return evaluator.run_evaluation()
