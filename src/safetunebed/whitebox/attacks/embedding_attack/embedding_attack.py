"""Schwinn et al. embedding attack (wraps around evaluator)."""

from dataclasses import dataclass

from typing_extensions import override

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.evals import (
    EmbeddingAttackEvaluation,
    EmbeddingAttackEvaluationConfig,
)
from safetunebed.whitebox.evals.embedding_attack import SoftOptConfig
from safetunebed.whitebox.utils.names import EvalName


@dataclass
class EmbeddingAttackConfig(TamperAttackConfig):
    """Hyper-parameters for an embedding attack.

    Attributes:
        soft_opt_config: Configuration of soft prompt optimization.
        small: Boolean flag on whether to use the small or large dataset for evaluation.
    """

    soft_opt_config: SoftOptConfig
    small: bool


class EmbeddingAttack(TamperAttack[EmbeddingAttackConfig]):
    """Embedding Attack class with functions to attack, predict and evaluate."""

    @override
    def run_attack(self) -> None:
        """Placehold attack function, no weights change for embedding attacks."""
        if (
            self.attack_config.base_input_checkpoint_path
            != self.attack_config.base_output_checkpoint_path
        ):
            msg = "There is no weight modification for embedding attacks, input and output ckpt paths should be same."
            raise ValueError(msg)

        if (
            self.attack_config.lora_input_folder
            != self.attack_config.lora_output_folder
        ):
            msg = "There is no weight modification for embedding attacks, input and output lora folders should be same."
            raise ValueError(msg)

    @override
    def evaluate(self) -> dict[str, float]:
        """Evaluate attack on relevant evaluations.

        Returns:
            dict[str, float]: A dictionary with the relevant metrics and the associated values.
        """
        results: dict[str, float] = {}

        if EvalName.EMBEDDING_ATTACK in self.attack_config.evals:
            results.update(self._evaluate_embedding_attack())

        return results

    def _evaluate_embedding_attack(self) -> dict[str, float]:
        """Evaluate attack on the `EmbeddingAttackEvaluation` evaluator."""
        eval_cfg: EmbeddingAttackEvaluationConfig = EmbeddingAttackEvaluationConfig(
            base_checkpoint=self.attack_config.base_output_checkpoint_path,
            lora_folder=self.attack_config.lora_output_folder,
            max_generation_length=self.attack_config.max_generation_length,
            small=self.attack_config.small,
            soft_opt_config=self.attack_config.soft_opt_config,
            batch_size=None,
        )
        evaluator: EmbeddingAttackEvaluation = EmbeddingAttackEvaluation(eval_cfg)

        return evaluator.run_evaluation()
