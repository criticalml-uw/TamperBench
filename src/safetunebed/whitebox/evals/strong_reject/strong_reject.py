"""StrongREJECT evaluator interface."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
from datasets.arrow_dataset import (
    Dataset as ArrowDataset,
)
from strong_reject import load_datasets as load_strong_reject_datasets
from strong_reject.evaluate import (
    strongreject_finetuned,
)
from torch import Tensor
from tqdm import tqdm
from transformers.generation.utils import (
    GenerateOutput,
)
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import override

from safetunebed.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from safetunebed.whitebox.utils import EvalName, dealloc_model_and_tokenizer

S = TypeVar(name="S", bound="StrongRejectEvaluationConfig")


@dataclass
class StrongRejectEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark.

    Attributes:
        small: Boolean flag on whether to use the small or large dataset.
        batch_size: Batch size for running inference
    """

    small: bool
    batch_size: int | None

    def __post_init__(self) -> None:
        """Set appropriate defaults where applicable."""
        self.batch_size = self.batch_size or 1


class StrongRejectEvaluation(WhiteBoxEvaluation[S]):
    """StrongREJECT Evaluation class."""

    name: EvalName = EvalName.STRONG_REJECT

    @override
    def run_inferences(self) -> dict[str, list[str]]:
        """Run inferences on the StrongReject dataset.

        Returns:
            dict[str, str]: A dictionary mapping LLM input and output for inferences.
        """
        model, tokenizer = self.load_model_and_tokenizer()
        strongreject_dataset: ArrowDataset = self._load_strong_reject_dataset()

        prompts: list[str] = [row["forbidden_prompt"] for row in strongreject_dataset]  # pyright: ignore[reportCallIssue]
        inferences: dict[str, list[str]] = {"forbidden_prompt": [], "response": []}

        for i in tqdm(range(0, len(prompts), self.eval_config.batch_size)):
            batch_prompts: list[str] = prompts[i : i + self.eval_config.batch_size]
            batch_encoding: BatchEncoding = tokenizer(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.eval_config.max_generation_length,
            )

            input_ids: Tensor = batch_encoding["input_ids"].cuda()
            attention_mask: Tensor = batch_encoding["attention_mask"].cuda()

            with torch.no_grad():
                batch_outputs: GenerateOutput = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.eval_config.max_generation_length,
                )

            for prompt, output_ids in zip(batch_prompts, batch_outputs, strict=False):  # pyright: ignore[reportUnknownArgumentType]
                text: str = tokenizer.decode(
                    token_ids=output_ids,  # pyright: ignore[reportUnknownArgumentType]
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                inferences["forbidden_prompt"].append(prompt)
                inferences["response"].append(text)

        dealloc_model_and_tokenizer(model, tokenizer)

        return inferences

    @override
    def run_evaluation(self) -> dict[str, float]:
        """Run evaluation and report metrics for the embedding attack evaluator (demo).

        Returns:
            dict[str, float]: A dictionary with the relevant evaluation metrics and the associated values.
        """
        inferences: dict[str, list[str]] = self.run_inferences()
        scores: dict[str, list[float]] = strongreject_finetuned(
            batch=inferences,
            max_response_length=self.eval_config.max_generation_length,
        )

        return {"strongreject_score": float(np.mean(scores["score"]))}

    def _load_strong_reject_dataset(self) -> ArrowDataset:
        """Load the appropriate version of the StrongReject dataset into an Arrow Dataset."""
        return (
            load_strong_reject_datasets.load_strongreject_small()
            if self.eval_config.small
            else load_strong_reject_datasets.load_strongreject()
        )
