"""Schwinn et al. input space embedding attack evaluator with JailBreakBench dataset."""  # TODO@<saad>: Adapt for StrongReject and appropriate datasets

# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false, reportUnknownArgumentType=false, reportIndexIssue=false, reportMissingTypeStubs=false, reportUnknownMemberType=false

from dataclasses import dataclass

import torch
from datasets import (
    DatasetDict,
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from torch._tensor import Tensor
from tqdm import tqdm
from transformers.generation.utils import GenerateOutput
from typing_extensions import override

from safetunebed.whitebox.evals.embedding_attack.softopt import (
    SoftOptConfig,
    SoftOptResult,
    run_soft_opt,
)
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.utils import dealloc_model_and_tokenizer
from safetunebed.whitebox.utils.names import EvalName


@dataclass
class EmbeddingAttackEvaluationConfig(
    StrongRejectEvaluationConfig
):  # TODO@<saad>: Adapt for StrongReject and appropriate datasets
    """Evaluation Configuration for Embedding Attack evaluation.

    Attributes:
        soft_opt_config: Configuration of soft prompt optimization.
    """

    soft_opt_config: SoftOptConfig


class EmbeddingAttackEvaluation(
    StrongRejectEvaluation[EmbeddingAttackEvaluationConfig],
):
    """EmbeddingAttackEvaluation class for an evaluation."""

    name: EvalName = EvalName.EMBEDDING_ATTACK

    @override
    def run_inferences(self) -> dict[str, list[str]]:
        """Run inferences on the JailBreak Behaviours dataset using latent embedding attack.

        Returns:
            dict[str, list[str]]: A dictionary mapping LLM input and output for inferences.
        """
        model, tokenizer = self.load_model_and_tokenizer()
        jbb_dataset: DatasetDict = (
            self._load_jbb_dataset()
        )  # TODO@<saad>: Adapt for StrongReject and appropriate datasets

        inferences: dict[str, list[str]] = {"forbidden_prompt": [], "response": []}
        for data_row in tqdm(jbb_dataset, total=len(jbb_dataset)):
            soft_opt_results: list[SoftOptResult] = run_soft_opt(
                model=model,
                tokenizer=tokenizer,
                messages=data_row["Goal"],
                target=data_row["Target"],
                config=self.eval_config.soft_opt_config,
            )

            for soft_opt_result in soft_opt_results:
                input_embeds: Tensor = soft_opt_result.input_embeds.to(
                    device=model.device, dtype=model.dtype
                )
                with torch.no_grad():
                    output: GenerateOutput = model.generate(  # pyright: ignore[reportUnknownVariableType]
                        inputs_embeds=input_embeds,
                        max_length=self.eval_config.max_generation_length,
                        output_hidden_states=False,
                    )
                    response_txt: str = tokenizer.decode(
                        token_ids=output[0], skip_special_tokens=True
                    ).strip()

                inferences["forbidden_prompt"].append(data_row["Goal"])
                inferences["response"].append(response_txt)

        dealloc_model_and_tokenizer(model, tokenizer)

        return inferences

    def _load_jbb_dataset(self) -> DatasetDict:
        """Load the JailBreakBehaviours dataset."""
        return load_dataset(  # pyright: ignore[reportReturnType]
            path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful"
        )
