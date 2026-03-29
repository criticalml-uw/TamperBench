"""SafetyGap evaluation for bio, chem, and cyber hazard propensity."""

from dataclasses import dataclass, field
from enum import Enum

import polars as pl
import torch
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig, load_vllm_model_and_tokenizer
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.utils import (
    aggregate_by_field,
    generate,
    strong_reject_score,
)
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
)
from tamperbench.whitebox.utils.datasets import get_dataset


class SafetyGapDomain(str, Enum):
    """Domains available in SafetyGap evaluation."""

    BIO = "bio"
    CHEM = "chem"
    CYBER = "cyber"


DATASET_URLS: dict[SafetyGapDomain, str] = {
    SafetyGapDomain.BIO: "https://raw.githubusercontent.com/AlignmentResearch/safety-gap/main/datasets/propensity_bio.csv",
    SafetyGapDomain.CHEM: "https://raw.githubusercontent.com/AlignmentResearch/safety-gap/main/datasets/propensity_chem.csv",
    SafetyGapDomain.CYBER: "https://raw.githubusercontent.com/AlignmentResearch/safety-gap/main/datasets/propensity_cyber.csv",
}

DOMAIN_METRICS: dict[SafetyGapDomain, MetricName] = {
    SafetyGapDomain.BIO: MetricName.SAFETY_GAP_BIO_SCORE,
    SafetyGapDomain.CHEM: MetricName.SAFETY_GAP_CHEM_SCORE,
    SafetyGapDomain.CYBER: MetricName.SAFETY_GAP_CYBER_SCORE,
}


@dataclass
class SafetyGapEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the SafetyGap benchmark.

    Attributes:
        domains: List of domains to evaluate. Defaults to all domains.
    """

    domains: list[SafetyGapDomain] = field(default_factory=lambda: list(SafetyGapDomain))


@register_evaluation(EvalName.SAFETY_GAP, SafetyGapEvaluationConfig)
class SafetyGapEvaluation(WhiteBoxEvaluation[SafetyGapEvaluationConfig]):
    """SafetyGap Evaluation class.

    Evaluates model propensity to generate harmful content across bio, chem, and cyber domains.
    Uses StrongReject classifier for scoring responses.
    """

    name: EvalName = EvalName.SAFETY_GAP
    objective: MetricName = MetricName.SAFETY_GAP_SCORE
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    # No reference_scores: The SafetyGap paper (Chua et al., 2025) does not report
    # StrongReject scores for baseline models, so there are no published values to
    # validate against. Uses the default empty ReferenceScores() from the base class.

    def _load_prompts_with_domains(self) -> list[dict[str, str]]:
        """Load prompts from configured domains only."""
        return [
            {
                "question": row["questions"],
                "domain": domain.value,
            }
            for domain in self.eval_config.domains
            for row in get_dataset(DATASET_URLS[domain], data_format="csv")
        ]

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on SafetyGap dataset to compute inferences."""
        prompts_with_domains = self._load_prompts_with_domains()

        prompts = [p["question"] for p in prompts_with_domains]
        domains = [p["domain"] for p in prompts_with_domains]

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, prompts),
            error_context="SafetyGap inference",
        )

        # Add domain column back (not passed through isolation to keep it simple)
        payload = payload.with_columns(pl.Series("domain", domains))

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Run StrongReject evaluator on model inferences."""
        return strong_reject_score(
            inferences,
            batch_size=self.eval_config.batch_size,
            max_response_length=self.eval_config.model_config.max_generation_length,
            extra_fields=["domain"],
        )

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate scores into per-domain and overall metrics."""
        # Get per-domain metrics
        metrics = aggregate_by_field(scores, "domain")

        metric_names: list[str] = []
        metric_values: list[float] = []

        # Map domain names to metric names
        for domain in self.eval_config.domains:
            if domain.value in metrics:
                metric_names.append(str(DOMAIN_METRICS[domain]))
                metric_values.append(metrics[domain.value])

        # Add overall score
        metric_names.append(str(SafetyGapEvaluation.objective))
        metric_values.append(metrics["overall"])

        return EvaluationSchema.validate(
            pl.from_dict(
                {
                    EvaluationSchema.metric_name: metric_names,
                    "metric_value": metric_values,
                }
            )
        )


def _instantiate_model_and_infer(
    eval_config: SafetyGapEvaluationConfig,
    prompts: list[str],
) -> pl.DataFrame:
    """Run vLLM inference on SafetyGap prompts.

    Prompts are passed raw (no chat template formatting). The SafetyGap paper
    evaluates propensity using raw completion prompts.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        prompts: List of raw prompt strings.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    model, _ = load_vllm_model_and_tokenizer(eval_config)
    try:
        responses = generate(
            prompts,
            model,
            max_new_tokens=eval_config.model_config.max_generation_length,
        )

        return pl.from_dict(
            {
                InferenceSchema.prompt: prompts,
                InferenceSchema.response: responses,
            }
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
