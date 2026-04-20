"""Prompt attack evaluator using StrongReject dataset."""

from dataclasses import dataclass

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.evals.base import load_tokenizer
from tamperbench.whitebox.evals.output_schema import InferenceSchema
from tamperbench.whitebox.evals.prompt_attack.dictionary import templates
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
    instantiate_model_and_infer,
)
from tamperbench.whitebox.utils.models.chat_format import format_chat_prompt
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
        tokenizer = load_tokenizer(self.eval_config)
        raw_prompts: list[str] = self.load_strong_reject_prompts()

        prompt_prefix = templates[self.eval_config.jailbreak_prompt]
        jailbreak_prompts = [format_chat_prompt(f"{prompt_prefix}{prompt}", tokenizer) for prompt in raw_prompts]

        payload: pl.DataFrame = run_in_isolation(
            target=instantiate_model_and_infer,
            args=(self.eval_config, jailbreak_prompts),
            error_context="Prompt attack inference",
        )

        # Replace jailbreak prompts with original forbidden prompts for StrongReject scoring
        payload = payload.with_columns(pl.Series(InferenceSchema.prompt, raw_prompts))
        return InferenceSchema.validate(payload)
