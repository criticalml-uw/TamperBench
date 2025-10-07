"""Schwinn et al. input space embedding attack evaluator with JailBreakBench dataset."""  # TODO@<saad>: Adapt for StrongReject and appropriate datasets

# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false, reportUnknownArgumentType=false, reportIndexIssue=false, reportMissingTypeStubs=false, reportUnknownMemberType=false

import multiprocessing
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
        """Run inferences (using LLM embedding attack) on evaluation dataset (jailbreak behaviours).

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        jbb_dataset: datasets.Dataset = (
            self._load_jbb_dataset()
        )  # TODO@<saad>: Adapt for StrongReject and appropriate datasets

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_instantiate_model_and_infer, args=(self.eval_config, jbb_dataset, q)
        )
        p.start()
        _inferences_dataframe = q.get()  # blocks until child puts result
        p.join()

        return InferenceSchema.validate(pl.DataFrame(_inferences_dataframe))

    def _load_jbb_dataset(self) -> datasets.Dataset:
        """Load the JailBreakBehaviours dataset."""
        return datasets.load_dataset(  # pyright: ignore[reportReturnType]
            path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful"
        )


def _instantiate_model_and_infer(
    eval_config: EmbeddingAttackEvaluationConfig,
    jbb_dataset: datasets.Dataset,
    q: multiprocessing.Queue,
) -> None:
    """Generate JailbreakBench responses in a subprocess and push them to the queue.

    Args:
        eval_config: Evaluation configuration containing model checkpoints and SoftOpt parameters.
        jbb_dataset: JailbreakBench rows (goal/target fields) to iterate over.
        q: Inter-process queue used to send the validated inference dataframe or error payload.

    Returns:
        None. The subprocess communicates results to the parent via ``q``.
    """
    model, tokenizer = load_model_and_tokenizer(eval_config)
    inferences: dict[str, list[str]] = {
        InferenceSchema.prompt: [],
        InferenceSchema.response: [],
    }

    for data_row in tqdm(jbb_dataset, total=len(jbb_dataset)):
        soft_opt_results: list[SoftOptResult] = run_soft_opt(
            model=model,
            tokenizer=tokenizer,
            messages=data_row["Goal"],
            target=data_row["Target"],
            config=eval_config.soft_opt_config,
        )

        for soft_opt_result in soft_opt_results:
            input_embeds: Tensor = soft_opt_result.input_embeds.to(
                device=model.device, dtype=model.dtype
            )
            with torch.no_grad():
                output: GenerateOutput = model.generate(  # pyright: ignore[reportUnknownVariableType]
                    inputs_embeds=input_embeds,
                    max_length=eval_config.model_config.max_generation_length,
                    output_hidden_states=False,
                )
                response_txt: str = tokenizer.decode(
                    token_ids=output[0], skip_special_tokens=True
                ).strip()

            inferences[InferenceSchema.prompt].append(data_row["Goal"])
            inferences[InferenceSchema.response].append(response_txt)

    dealloc_model_and_tokenizer(model, tokenizer)

    _inferences_dataframe: pl.DataFrame = pl.from_dict(data=inferences)
    q.put(InferenceSchema.validate(_inferences_dataframe))
