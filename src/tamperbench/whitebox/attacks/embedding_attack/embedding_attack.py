"""Schwinn et al. embedding attack (wraps around evaluator)."""

# pyright: reportUnknownMemberType=false, reportImplicitOverride=false

from dataclasses import dataclass
from typing import Any

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals import (
    EmbeddingAttackEvaluation,
    EmbeddingAttackEvaluationConfig,
)
from tamperbench.whitebox.evals.embedding_attack import SoftOptConfig
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.names import AttackName, EvalName


@dataclass
class EmbeddingAttackConfig(TamperAttackConfig):
    """Hyper-parameters for an embedding attack.

    Attributes:
        soft_opt_config: Configuration of soft prompt optimization.
    """

    soft_opt_config: SoftOptConfig

    @classmethod
    def prepare_dict_for_init(cls, data: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Transform dictionary data into proper types for dataclass initialization.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            dict[str, Any]: Dictionary with nested objects properly instantiated.
        """
        data = super().prepare_dict_for_init(data)

        soft_opt_config_dict = data.pop("soft_opt_config")  # pyright: ignore[reportAny]

        # Pass random_seed from attack config to SoftOptConfig
        soft_opt_config_dict["seed"] = data["random_seed"]
        soft_opt_config = SoftOptConfig(**soft_opt_config_dict)  # pyright: ignore[reportAny]

        data.update({"soft_opt_config": soft_opt_config})
        return data


@register_attack(AttackName.EMBEDDING_ATTACK, EmbeddingAttackConfig)
class EmbeddingAttack(TamperAttack[EmbeddingAttackConfig]):
    """Embedding Attack class with functions to attack, predict and evaluate."""

    name: AttackName = AttackName.EMBEDDING_ATTACK

    @override
    def run_attack(self) -> None:
        """Placehold attack function, no weights change for embedding attacks."""
        pass

    @override
    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on relevant evaluations.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        results = EvaluationSchema.empty()

        if EvalName.EMBEDDING_ATTACK in self.attack_config.evals:
            results = pl.concat(items=[results, self.evaluate_embedding_attack()])

        return EvaluationSchema.validate(results)

    def evaluate_embedding_attack(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `EmbeddingAttackEvaluation` evaluator."""
        eval_cfg: EmbeddingAttackEvaluationConfig = EmbeddingAttackEvaluationConfig(
            model_checkpoint=self.attack_config.input_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            soft_opt_config=self.attack_config.soft_opt_config,
        )
        evaluator: EmbeddingAttackEvaluation = EmbeddingAttackEvaluation(eval_cfg)

        return evaluator.run_evaluation()
