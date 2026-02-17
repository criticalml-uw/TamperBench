"""StrongREJECT evaluator interface."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportOptionalMemberAccess=false

import logging
import multiprocessing
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

import datasets
import polars as pl
import torch
from datasets.arrow_dataset import Dataset as ArrowDataset
from pandera.typing.polars import DataFrame
from polars.series.series import Series
from strong_reject import load_datasets as load_strong_reject_datasets
from strong_reject.evaluate import strongreject_finetuned as strong_reject_finetuned
from strong_reject.evaluate import strongreject_rubric
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from typing_extensions import override
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
)
from tamperbench.whitebox.utils.ops import run_in_isolation

logger = logging.getLogger(__name__)

S = TypeVar(name="S", bound="StrongRejectEvaluationConfig")

multiprocessing.set_start_method("spawn", force=True)  # TODO: check if there is alternate non-global config for this


@dataclass
class StrongRejectEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark.

    Attributes:
        max_samples: Maximum number of samples to evaluate. If None, use full dataset.
    """

    max_samples: int | None = field(default=None, kw_only=True)


@dataclass
class StrongRejectRubricEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for the StrongReject rubric-based evaluator.

    Attributes:
        judge_models: LLM models for the rubric judge. If None, uses package defaults
            (``gpt-4o-mini``, ``gpt-3.5-turbo``).
        max_workers: Thread pool size for parallel API calls.
    """

    judge_models: list[str] | None = field(default=None, kw_only=True)
    max_workers: int = field(default=16, kw_only=True)


class _StrongRejectBaseEvaluation(WhiteBoxEvaluation[S], ABC):
    """Base class with shared inference and prompt-loading logic for StrongREJECT evaluations."""

    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate evaluator outputs into overall evaluation metrics.

        Args:
            scores: A polars DataFrame with the model inferences scored by evaluator.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the metrics and scores of the full evaluation.
        """
        scores_dataframe: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        mean_score: float = float(
            scores_dataframe[ScoreSchema.score].mean()  # pyright: ignore[reportUnknownArgumentType]
        )
        _metrics_dataframe: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [str(type(self).objective)],
                EvaluationSchema.metric_value: [mean_score],
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on evaluation dataset (StrongREJECT) to compute the inferences.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        prompts: list[str] = self.load_strong_reject_prompts()

        payload: pl.DataFrame = run_in_isolation(
            target=instantiate_model_and_infer,
            args=(self.eval_config, prompts),
            error_context="StrongReject inference",
        )

        return InferenceSchema.validate(payload)

    def load_strong_reject_prompts(self) -> list[str]:
        """Load the StrongReject dataset into an Arrow Dataset, and then return prompts.

        Returns:
            list[str]: A list of prompts from the StrongReject dataset to input to the model to obtain inferences.
        """
        strong_reject_dataset: ArrowDataset = load_strong_reject_datasets.load_strongreject()
        if self.eval_config.max_samples is not None:
            strong_reject_dataset = strong_reject_dataset.select(range(self.eval_config.max_samples))

        return [row["forbidden_prompt"] for row in strong_reject_dataset]  # pyright: ignore[reportCallIssue]


@register_evaluation(EvalName.STRONG_REJECT_FINETUNED)
class StrongRejectFinetunedEvaluation(_StrongRejectBaseEvaluation[S]):
    """StrongREJECT Evaluation class using the fine-tuned classifier."""

    name: EvalName = EvalName.STRONG_REJECT_FINETUNED
    objective: MetricName = MetricName.STRONG_REJECT_FINETUNED_SCORE

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Run evaluator (StrongREJECT fine-tuned) on model inferences to compute evaluation scores on data points.

        Args:
            inferences: A polars DataFrame with the prompts to LLM and corresponding responses.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored by evaluator.
        """
        inferences_dataframe: DataFrame[InferenceSchema] = InferenceSchema.validate(inferences)

        payload: pl.DataFrame = run_in_isolation(
            target=_compute_strong_reject_finetuned_scores,
            args=(self.eval_config, inferences_dataframe.to_dict()),
            error_context="StrongReject finetuned scoring",
        )

        return ScoreSchema.validate(payload)


_RUBRIC_MAX_RETRIES = 3
_RUBRIC_RETRY_BACKOFF_SECONDS = 2.0


@register_evaluation(EvalName.STRONG_REJECT)
class StrongRejectEvaluation(_StrongRejectBaseEvaluation[StrongRejectRubricEvaluationConfig]):
    """StrongREJECT Evaluation class using the rubric-based LLM judge.

    Not generic — always requires ``StrongRejectRubricEvaluationConfig`` (unlike the
    finetuned variant which is subclassed by GCG, EmbeddingAttack, etc.).
    """

    name: EvalName = EvalName.STRONG_REJECT
    objective: MetricName = MetricName.STRONG_REJECT_SCORE

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Run evaluator (StrongREJECT rubric) on model inferences to compute evaluation scores.

        Calls the ``strongreject_rubric`` evaluator per-instance, parallelized via thread pool
        (API calls, no GPU isolation needed).  Each call is retried up to
        ``_RUBRIC_MAX_RETRIES`` times with exponential back-off on transient failures.

        Args:
            inferences: A polars DataFrame with the prompts to LLM and corresponding responses.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored by evaluator.
        """
        inferences_dataframe: DataFrame[InferenceSchema] = InferenceSchema.validate(inferences)

        prompts: list[str] = inferences_dataframe[InferenceSchema.prompt].to_list()
        responses: list[str] = inferences_dataframe[InferenceSchema.response].to_list()

        rubric_kwargs: dict[str, object] = {}
        if self.eval_config.judge_models is not None:
            rubric_kwargs["models"] = self.eval_config.judge_models

        def score_row(args: tuple[str, str]) -> float:
            for attempt in range(_RUBRIC_MAX_RETRIES):
                try:
                    result = strongreject_rubric(forbidden_prompt=args[0], response=args[1], **rubric_kwargs)
                    return result["score"]
                except Exception:
                    if attempt == _RUBRIC_MAX_RETRIES - 1:
                        raise
                    wait = _RUBRIC_RETRY_BACKOFF_SECONDS * (2**attempt)
                    logger.warning(
                        "strongreject_rubric failed (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        _RUBRIC_MAX_RETRIES,
                        wait,
                        exc_info=True,
                    )
                    time.sleep(wait)
            raise AssertionError("unreachable")  # pragma: no cover

        scores: list[float] = thread_map(
            score_row,
            list(zip(prompts, responses, strict=True)),
            max_workers=self.eval_config.max_workers,
            desc="StrongREJECT rubric scoring",
        )

        return ScoreSchema.validate(
            pl.from_dict(
                data={
                    InferenceSchema.prompt: prompts,
                    InferenceSchema.response: responses,
                    ScoreSchema.score: scores,
                }
            )
        )


@register_evaluation(EvalName.JAILBREAK_BENCH)
class JailbreakBenchEvaluation(StrongRejectFinetunedEvaluation[S]):
    """JailbreakBench Evaluation class using the JBB-Behaviors dataset, scored by StrongREJECT."""

    name: EvalName = EvalName.JAILBREAK_BENCH
    objective: MetricName = MetricName.JAILBREAK_BENCH_STRONG_REJECT_SCORE

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the JailbreakBench JBB-Behaviors dataset and return prompts.

        Returns:
            list[str]: A list of prompts from the JailbreakBench dataset to input to the model to obtain inferences.
        """
        jbb_dataset = datasets.load_dataset(  # pyright: ignore[reportReturnType]
            path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful"
        )

        user_prefix = self.eval_config.model_config.user_prefix
        assistant_prefix = self.eval_config.model_config.assistant_prefix
        end_turn = self.eval_config.model_config.end_turn

        def to_text(data_row: dict[str, str]) -> str:
            return f"{user_prefix}{data_row['Goal']}{end_turn}{assistant_prefix}"

        return [to_text(row) for row in jbb_dataset]  # pyright: ignore[reportCallIssue]


def instantiate_model_and_infer(
    eval_config,
    prompts,
) -> pl.DataFrame:
    """Run model on evaluation dataset (StrongREJECT) to compute the inferences.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        prompts: List of prompts to generate responses for.

    Returns:
        A polars DataFrame with the prompts and corresponding model responses.
    """
    llm: LLM | None = None
    try:
        # vllm tensor_parallel_size must be a factor of 64 (model hidden dimensions).
        # TODO: Handle scaling beyond 4 GPUs properly by validating against model config.
        llm_kwargs = {
            "model": eval_config.model_checkpoint,
            "tensor_parallel_size": torch.cuda.device_count() if torch.cuda.device_count() != 5 else 4,
            "gpu_memory_utilization": 0.8,
            "trust_remote_code": True,
        }

        # Use separate tokenizer if specified
        if eval_config.model_config.tokenizer_checkpoint is not None:
            llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=eval_config.model_config.max_generation_length,
        )
        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        request_outputs = llm.generate(prompts, sampling_params)

        for prompt, request_output in zip(prompts, request_outputs, strict=False):  # pyright: ignore[reportUnknownArgumentType]
            text: str = ""
            if request_output.outputs:
                text = request_output.outputs[0].text.strip()

            inferences[InferenceSchema.prompt].append(prompt)
            inferences[InferenceSchema.response].append(text)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compute_strong_reject_finetuned_scores(eval_config, inferences_dict) -> pl.DataFrame:
    """Run StrongReject finetuned scoring inside an isolated process to release GPU memory cleanly.

    Args:
        eval_config: Evaluation configuration with model settings.
        inferences_dict: Dictionary representation of the inferences DataFrame.

    Returns:
        A polars DataFrame with the scored inferences.
    """
    inferences_dataframe = pl.from_dict(data=inferences_dict)

    scores_dict: dict[str, Series] = inferences_dataframe.to_dict()  # initialize scores with inferences
    scores: list[float] = []

    for index in tqdm(
        range(
            0,
            len(inferences_dataframe),
            eval_config.model_config.inference_batch_size,
        )
    ):
        batch_inferences: dict[str, list[str]] = {
            "forbidden_prompt": list(
                inferences_dataframe[InferenceSchema.prompt][
                    index : index + eval_config.model_config.inference_batch_size
                ]
            ),
            "response": list(
                inferences_dataframe[InferenceSchema.response][
                    index : index + eval_config.model_config.inference_batch_size
                ]
            ),
        }

        scores.extend(
            strong_reject_finetuned(
                batch=batch_inferences,
                max_response_length=eval_config.model_config.max_generation_length,
            ).get("score")
        )

    scores_dict.update({ScoreSchema.score: pl.Series(scores)})

    return ScoreSchema.validate(pl.from_dict(data=scores_dict))
