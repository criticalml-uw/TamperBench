"""StrongREJECT evaluator interface."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false

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

from safetunebed.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from safetunebed.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from safetunebed.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
)

S = TypeVar(name="S", bound="StrongRejectEvaluationConfig")


USER_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>\n"

# USER_PREFIX       = "[INST] "      # note the trailing space
# ASSISTANT_PREFIX  = " [/INST] "    # note the leading & trailing spaces
# END_TURN          = "</s>"         # or: tokenizer.eos_token


@dataclass
class StrongRejectEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark."""


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
        model, tokenizer = self.load_model_and_tokenizer()
        prompts: list[str] = self.load_strong_reject_prompts()

        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        for i in tqdm(range(0, len(prompts), self.eval_config.batch_size)):
            batch_prompts: list[str] = prompts[i : i + self.eval_config.batch_size]  # pyright: ignore[reportOperatorIssue]
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
                    max_length=self.eval_config.max_generation_length,
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

        _inferences_dataframe: pl.DataFrame = pl.from_dict(data=inferences)
        return InferenceSchema.validate(_inferences_dataframe)

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

        scores_dict: dict[str, Series] = (
            inferences_dataframe.to_dict()
        )  # initialize scores with inferences
        scores: list[float] = []

        for i in tqdm(range(0, len(inferences_dataframe), self.eval_config.batch_size)):
            batch_inferences: dict[str, list[str]] = {
                "forbidden_prompt": list(
                    inferences_dataframe[InferenceSchema.prompt][
                        i : i + self.eval_config.batch_size
                    ]
                ),
                "response": list(
                    inferences_dataframe[InferenceSchema.response][
                        i : i + self.eval_config.batch_size
                    ]
                ),
            }

            scores.extend(
                strong_reject_finetuned(
                    batch=batch_inferences,
                    max_response_length=self.eval_config.max_generation_length,
                ).get("score")
            )

        scores_dict.update({ScoreSchema.score: pl.Series(scores)})

        _scores_dataframe: pl.DataFrame = pl.from_dict(data=scores_dict)
        return ScoreSchema.validate(_scores_dataframe)

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

        def to_text(ex):  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
            return f"{USER_PREFIX}{ex['forbidden_prompt']}{END_TURN}{ASSISTANT_PREFIX}"

        return [to_text(row) for row in strong_reject_dataset]  # pyright: ignore[reportCallIssue]
