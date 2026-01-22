"""Schwinn et al. input space embedding attack evaluator with JailBreakBench dataset."""  # TODO@<saad>: Adapt for StrongReject and appropriate datasets

# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from dataclasses import dataclass

import datasets
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from torch._tensor import Tensor
from tqdm import tqdm
from transformers.generation.utils import GenerateOutput
from typing_extensions import override

from safetunebed.whitebox.evals.base import load_model_and_tokenizer
from safetunebed.whitebox.evals.embedding_attack.softopt import (
    SoftOptConfig,
    SoftOptResult,
    run_soft_opt,
)
from safetunebed.whitebox.evals.output_schema import InferenceSchema
from safetunebed.whitebox.evals.registry import register_evaluation
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.utils import dealloc_model_and_tokenizer
from safetunebed.whitebox.utils.names import EvalName
from safetunebed.whitebox.utils.ops import run_in_isolation


@dataclass
class EmbeddingAttackEvaluationConfig(
    StrongRejectEvaluationConfig
):  # TODO@<saad>: Adapt for StrongReject and appropriate datasets
    """Evaluation Configuration for Embedding Attack evaluation.

    Attributes:
        soft_opt_config: Configuration of soft prompt optimization.
    """

    soft_opt_config: SoftOptConfig


@register_evaluation(EvalName.EMBEDDING_ATTACK)
class EmbeddingAttackEvaluation(
    StrongRejectEvaluation[EmbeddingAttackEvaluationConfig],
):
    """EmbeddingAttackEvaluation class for an evaluation."""

    name: EvalName = EvalName.EMBEDDING_ATTACK

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run inferences (using LLM embedding attack) on evaluation dataset (StrongReject).

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        strongreject_dataset: datasets.DatasetDict = self._load_strongreject_dataset()

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, strongreject_dataset),
            error_context="Embedding attack inference",
        )

        return InferenceSchema.validate(payload)

    def _load_strongreject_dataset(self) -> datasets.DatasetDict:
        """Load the StrongReject dataset with targets."""
        return datasets.load_dataset(  # pyright: ignore[reportReturnType]
            path="sdhossain24/strongreject_with_targets", split="train"
        )


def _instantiate_model_and_infer(
    eval_config,
    strongreject_dataset,
) -> pl.DataFrame:
    """Run model on evaluation dataset (StrongREJECT) to compute inferences.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        strongreject_dataset: StrongReject dataset with prompts and targets.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    model, tokenizer = load_model_and_tokenizer(eval_config)
    try:
        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        for data_row in tqdm(strongreject_dataset, total=len(strongreject_dataset)):
            soft_opt_results: list[SoftOptResult] = run_soft_opt(
                model=model,
                tokenizer=tokenizer,
                messages=data_row["prompt"],
                target=data_row["target"],
                config=eval_config.soft_opt_config,
            )

            for soft_opt_result in soft_opt_results:
                input_embeds: Tensor = soft_opt_result.input_embeds.to(device=model.device, dtype=model.dtype)
                with torch.no_grad():
                    output: GenerateOutput = model.generate(
                        inputs_embeds=input_embeds,
                        max_length=eval_config.model_config.max_generation_length,
                        output_hidden_states=False,
                    )
                    response_txt: str = tokenizer.decode(token_ids=output[0], skip_special_tokens=True).strip()

                inferences[InferenceSchema.prompt].append(data_row["prompt"])
                inferences[InferenceSchema.response].append(response_txt)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        dealloc_model_and_tokenizer(model, tokenizer)
