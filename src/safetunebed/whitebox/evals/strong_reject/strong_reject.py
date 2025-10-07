"""StrongREJECT evaluator interface."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false

import multiprocessing
from dataclasses import dataclass
from typing import TypeVar

import polars as pl
import torch
from datasets.arrow_dataset import (
    Dataset as ArrowDataset,
)
from pandera.typing.polars import DataFrame
from polars.series.series import Series
from strong_reject import load_datasets as load_strong_reject_datasets
from strong_reject.evaluate import strongreject_finetuned as strong_reject_finetuned
from torch import Tensor
from tqdm import tqdm
from transformers.generation.utils import (
    GenerateOutput,
)
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import override

from safetunebed.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
    load_model_and_tokenizer,
)
from safetunebed.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from safetunebed.whitebox.evals.registry import register_evaluation
from safetunebed.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
)

S = TypeVar(name="S", bound="StrongRejectEvaluationConfig")

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class StrongRejectEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark."""


@register_evaluation(EvalName.STRONG_REJECT)
class StrongRejectEvaluation(WhiteBoxEvaluation[S]):
    """StrongREJECT Evaluation class."""

    name: EvalName = EvalName.STRONG_REJECT
    objective: MetricName = MetricName.STRONG_REJECT_SCORE
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on evaluation dataset (StrongREJECT) to compute the inferences.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        prompts: list[str] = self.load_strong_reject_prompts()

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_instantiate_model_and_infer, args=(self.eval_config, prompts, q)
        )
        p.start()
        try:
            child_message = q.get()
        finally:
            p.join()
            p.close()
            q.close()
            q.join_thread()

        if child_message.get("status") != "ok":
            error_message = child_message.get(
                "error", "Unknown error in StrongReject inference process."
            )
            stack_trace = child_message.get("traceback", "")
            raise RuntimeError(
                f"StrongReject inference failed in subprocess: {error_message}\n{stack_trace}"
            )

        payload = child_message.get("payload")
        if payload is None:
            raise RuntimeError("StrongReject inference subprocess returned no results.")
        if isinstance(payload, pl.DataFrame):
            return InferenceSchema.validate(payload)

        return InferenceSchema.validate(pl.DataFrame(payload))

    @override
    def compute_scores(
        self, inferences: DataFrame[InferenceSchema]
    ) -> DataFrame[ScoreSchema]:
        """Run evaluator (StrongREJECT) on model inferences to compute evaluation scores on data points.

        Args:
            inferences: A polars DataFrame with the prompts to LLM and corresponding responses.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored by evaluator.
        """
        inferences_dataframe: DataFrame[InferenceSchema] = InferenceSchema.validate(
            inferences
        )

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_compute_strong_reject_scores,
            args=(
                self.eval_config,
                inferences_dataframe.to_dict(),
                q,
            ),
        )
        p.start()
        try:
            child_message = q.get()
        finally:
            p.join()
            p.close()
            q.close()
            q.join_thread()

        if child_message.get("status") != "ok":
            error_message = child_message.get(
                "error", "Unknown error in StrongReject scoring process."
            )
            stack_trace = child_message.get("traceback", "")
            raise RuntimeError(
                f"StrongReject scoring failed in subprocess: {error_message}\n{stack_trace}"
            )

        payload = child_message.get("payload")
        if payload is None:
            raise RuntimeError("StrongReject scoring subprocess returned no results.")
        if isinstance(payload, pl.DataFrame):
            return ScoreSchema.validate(payload)

        return ScoreSchema.validate(pl.DataFrame(payload))

    @override
    def compute_results(
        self, scores: DataFrame[ScoreSchema]
    ) -> DataFrame[EvaluationSchema]:
        """Aggregate evaluator outputs into overall evaluation metrics.

        Takes the per-datapoint evaluator scores and compiles them into higher-level metrics that
        characterize the model's performance across the full evaluation dataset.

        Args:
            scores: A polars DataFrame with the model inferences scored by evaluator.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the metrics and scores of the full evaluation.
        """
        scores_dataframe: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        mean_strong_reject_score: float = float(
            scores_dataframe[ScoreSchema.score].mean()  # pyright: ignore[reportUnknownArgumentType]
        )
        _metrics_dataframe: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [str(StrongRejectEvaluation.objective)],
                EvaluationSchema.metric_value: [mean_strong_reject_score],
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)

    def load_strong_reject_prompts(self) -> list[str]:
        """Load the StrongReject dataset into an Arrow Dataset, and then return prompts.

        Returns:
            list[str]: A list of prompts from the StrongReject dataset to input to the model to obtain inferences.
        """
        strong_reject_dataset: ArrowDataset = (
            load_strong_reject_datasets.load_strongreject()
        )

        return [row["forbidden_prompt"] for row in strong_reject_dataset]  # pyright: ignore[reportCallIssue]


@register_evaluation(EvalName.STRONG_REJECT_SMALL)
class StrongRejectSmallEvaluation(StrongRejectEvaluation[S]):
    """StrongREJECT Evaluation class using a small version of the StrongREJECT dataset."""

    name: EvalName = EvalName.STRONG_REJECT_SMALL

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the small version of the StrongReject dataset into an Arrow Dataset, and then return prompts.

        Returns:
            list[str]: A list of prompts from the StrongReject dataset to input to the model to obtain inferences.
        """
        strong_reject_dataset: ArrowDataset = (
            load_strong_reject_datasets.load_strongreject_small()
        )

        user_prefix = self.eval_config.model_config.user_prefix
        assistant_prefix = self.eval_config.model_config.assistant_prefix
        end_turn = self.eval_config.model_config.end_turn

        def to_text(data_row: str) -> str:
            return f"{user_prefix}{data_row['forbidden_prompt']}{end_turn}{assistant_prefix}"

        return [to_text(row) for row in strong_reject_dataset]  # pyright: ignore[reportCallIssue]


def _instantiate_model_and_infer(
    eval_config: StrongRejectEvaluationConfig,
    prompts: list[str],
    q: multiprocessing.Queue,
) -> None:
    """Generate StrongReject responses in a subprocess and queue the results.

    Args:
        eval_config: Evaluation configuration containing model checkpoints and generation knobs.
        prompts: StrongReject prompts to feed into the model.
        q: Inter-process queue used to surface either validated results or failure details.

    Returns:
        None. The subprocess deposits a dict payload on ``q`` containing the inference dataframe or an error.
    """
    try:
        model, tokenizer = load_model_and_tokenizer(eval_config)
        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        for index in tqdm(
            range(0, len(prompts), eval_config.model_config.inference_batch_size)
        ):
            batch_prompts: list[str] = prompts[
                index : index + eval_config.model_config.inference_batch_size
            ]  # pyright: ignore[reportOperatorIssue]
            batch_encoding: BatchEncoding = tokenizer(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            input_ids: Tensor = batch_encoding["input_ids"].cuda()
            attention_mask: Tensor = batch_encoding["attention_mask"].cuda()

            with torch.no_grad():
                batch_outputs: GenerateOutput = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=eval_config.model_config.max_generation_length,
                    max_new_tokens=None,
                )

            for prompt, output_ids in zip(batch_prompts, batch_outputs, strict=False):  # pyright: ignore[reportUnknownArgumentType]
                text: str = tokenizer.decode(
                    token_ids=output_ids,  # pyright: ignore[reportUnknownArgumentType]
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                inferences[InferenceSchema.prompt].append(prompt)
                inferences[InferenceSchema.response].append(text)

        dealloc_model_and_tokenizer(model, tokenizer)

        validated = InferenceSchema.validate(pl.from_dict(data=inferences))
        q.put({"status": "ok", "payload": validated})
    except Exception as error:  # pragma: no cover - propagated to parent
        import traceback

        q.put(
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )


def _compute_strong_reject_scores(
    eval_config: StrongRejectEvaluationConfig,
    inferences_dict: dict[str, Series],
    q: multiprocessing.Queue,
) -> None:
    """Score StrongReject generations in a subprocess and return the payload via queue.

    Args:
        eval_config: Evaluation configuration containing scoring knobs such as max generation length.
        inferences_dict: Dictionary representation of the inference dataframe (prompt/response columns).
        q: Inter-process queue used to send either the validated score dataframe or an error payload.

    Returns:
        None. Results are relayed back to the parent process through ``q``.
    """
    try:
        inferences_dataframe = pl.from_dict(data=inferences_dict)

        scores_dict: dict[str, Series] = (
            inferences_dataframe.to_dict()
        )  # initialize scores with inferences
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

        validated_scores: pl.DataFrame = ScoreSchema.validate(
            pl.from_dict(data=scores_dict)
        )
        q.put({"status": "ok", "payload": validated_scores})
    except Exception as error:  # pragma: no cover - surfaced to parent
        import traceback

        q.put(
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
