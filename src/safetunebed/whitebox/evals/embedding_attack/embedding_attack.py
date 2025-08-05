"""Template for evaluations in safetunebed that evaluate potency of attacks."""

from dataclasses import dataclass

import torch
from datasets import DatasetDict, load_dataset
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
from safetunebed.whitebox.utils.names import EvalName


@dataclass
class EmbeddingAttackEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for Embedding Attack evaluation.

    Attributes:
        soft_opt_config: Configuration of soft prompt optimization.
        eval_size: Number of datapoints from JailBreakBehaviours to evaluate on.
    """

    soft_opt_config: SoftOptConfig


class EmbeddingAttackEvaluation(
    StrongRejectEvaluation[EmbeddingAttackEvaluationConfig],
):
    """EmbeddingAttackEvaluation class for an evaluation."""

    name: EvalName = EvalName.EMBEDDING_ATTACK

    @override
    def run_inferences(self) -> dict[str, list[str]]:
        """Run inferences on the JailBreak Behaviours dataset using latent embedding attack..

        Returns:
            dict[str, list[str]]: A dictionary mapping LLM input and output for inferences.
        """
        model, tokenizer = self.load_model_and_tokenizer()
        jbb_dataset: DatasetDict = self._load_jbb_dataset()

        inferences: dict[str, list[str]] = {"forbidden_prompt": [], "response": []}
        for data_row in tqdm(jbb_dataset, total=len(jbb_dataset)):
            soft_opt_result: SoftOptResult = run_soft_opt(
                model=model,
                tokenizer=tokenizer,
                messages=data_row["Goal"],
                target=data_row["Target"],
                config=self.eval_config.soft_opt_config,
            )

            input_embeds: Tensor = soft_opt_result.input_embeds.to(
                device=model.device, dtype=model.dtype
            )
            with torch.no_grad():
                output: GenerateOutput = model.generate(
                    inputs_embeds=input_embeds,
                    max_length=self.eval_config.max_generation_length,
                    output_hidden_states=False,
                )
                response_txt: str = tokenizer.decode(
                    token_ids=output[0], skip_special_tokens=True
                ).strip()

            inferences["forbidden_prompt"].append(data_row["Goal"])
            inferences["response"].append(response_txt)

        return inferences

    def _load_jbb_dataset(self) -> DatasetDict:
        """Load the JailBreakBehaviours dataset."""
        return load_dataset(
            path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful"
        )
