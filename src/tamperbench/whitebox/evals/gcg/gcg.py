"""GCG attack evaluator with StrongREJECT dataset."""

from dataclasses import dataclass, field
from typing import Any, cast

import datasets
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from tamperbench.whitebox.evals.base import load_hf_model_and_tokenizer
from tamperbench.whitebox.evals.gcg.implementation import GCG, GCGConfig
from tamperbench.whitebox.evals.output_schema import InferenceSchema
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluationConfig,
    StrongRejectFinetunedEvaluation,
)
from tamperbench.whitebox.utils import dealloc_model_and_tokenizer
from tamperbench.whitebox.utils.names import EvalName
from tamperbench.whitebox.utils.ops import run_in_isolation


@dataclass
class GCGEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for GCG attack evaluation.

    Attributes:
        gcg_config: Configuration for GCG optimization.
    """

    gcg_config: GCGConfig = field(default_factory=GCGConfig)


@register_evaluation(EvalName.GCG_ATTACK_EVAL)
class GCGEvaluation(
    StrongRejectFinetunedEvaluation[GCGEvaluationConfig],
):
    """GCG attack evaluation class."""

    name: EvalName = EvalName.GCG_ATTACK_EVAL

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run inferences (using GCG attack) on evaluation dataset (StrongReject).

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        strongreject_dataset: datasets.Dataset = self._load_strongreject_dataset()

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, strongreject_dataset),
            error_context="GCG attack inference",
        )

        return InferenceSchema.validate(payload)

    def _load_strongreject_dataset(self) -> datasets.Dataset:
        """Load the StrongReject dataset with targets."""
        return datasets.load_dataset(path="sdhossain24/strongreject_with_targets", split="train")


def _instantiate_model_and_infer(
    eval_config: GCGEvaluationConfig,
    strongreject_dataset: datasets.Dataset,
) -> pl.DataFrame:
    """Run GCG optimization and model inference on StrongREJECT dataset.

    Args:
        eval_config: Evaluation configuration with model checkpoint and GCG settings.
        strongreject_dataset: StrongReject dataset with prompts and targets.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    model, tokenizer = load_hf_model_and_tokenizer(eval_config)
    try:
        gcg = GCG(model, tokenizer, eval_config.gcg_config)

        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        for data_row in tqdm(strongreject_dataset, total=len(strongreject_dataset)):
            row: dict[str, Any] = cast(dict[str, Any], data_row)
            result = gcg.run(messages=row["prompt"], target=row["target"])

            # Build the chat-templated prompt with the GCG suffix in the same
            # position it occupied during optimization (end of user message).
            chat_messages = [{"role": "user", "content": row["prompt"] + result.best_string}]
            templated_prompt = cast(
                str, tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            )
            input_ids = cast(
                Tensor,
                tokenizer(
                    templated_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"],
            ).to(model.device)

            with torch.no_grad():
                output: Tensor = model.generate(  # pyright: ignore[reportCallIssue]
                    input_ids=input_ids,
                    max_new_tokens=eval_config.model_config.max_generation_length,
                )
                # Decode only the generated tokens (exclude input)
                generated_ids = output[0][input_ids.shape[1] :]
                response_txt: str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            inferences[InferenceSchema.prompt].append(row["prompt"])
            inferences[InferenceSchema.response].append(response_txt)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        dealloc_model_and_tokenizer(model, tokenizer)
