"""Prompt attack evaluator using StrongReject dataset."""

from dataclasses import dataclass

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.evals.output_schema import InferenceSchema
from tamperbench.whitebox.evals.prompt_attack.dictionary import templates
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
    instantiate_model_and_infer,
)
from tamperbench.whitebox.utils.names import EvalName
from tamperbench.whitebox.utils.ops import run_in_isolation


@dataclass
class PromptAttackEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for prompt attack."""

    jailbreak_prompt: str = "aim"


@register_evaluation(EvalName.PROMPT_ATTACK)
class PromptAttackEvaluation(StrongRejectEvaluation[PromptAttackEvaluationConfig]):
    """Prompt attack evaluator using StrongReject dataset."""

    name: EvalName = EvalName.PROMPT_ATTACK

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run inferences on the StrongReject dataset using vLLM.

        Prepends the jailbreak template to each prompt before sending to the model,
        then replaces the prompt column with the original forbidden prompts so that
        StrongReject scoring receives the correct inputs.
        """
        prompts: list[str] = self.load_strong_reject_prompts()

        prompt_prefix = templates[self.eval_config.jailbreak_prompt]
        jailbreak_prompts = [f"{prompt_prefix}{prompt}" for prompt in prompts]

        payload: pl.DataFrame = run_in_isolation(
            target=instantiate_model_and_infer,
            args=(self.eval_config, jailbreak_prompts),
            error_context="Prompt attack inference",
        )

        # Replace jailbreak prompts with original forbidden prompts for StrongReject scoring
        payload = payload.with_columns(pl.Series(InferenceSchema.prompt, prompts))
        return InferenceSchema.validate(payload)
