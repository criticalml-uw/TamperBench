"""Minerva MATH evaluator interface.

Based on lm-evaluation-harness implementation:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportMissingTypeArgument=false, reportOptionalMemberAccess=false

import multiprocessing
from dataclasses import dataclass

import datasets
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from typing_extensions import override
from vllm import LLM, SamplingParams

from safetunebed.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from safetunebed.whitebox.evals.minerva_math import utils
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
)
from safetunebed.whitebox.utils.ops import run_in_isolation

multiprocessing.set_start_method("spawn", force=True)

DATASET_NAME = "EleutherAI/hendrycks_math"
# Combine all subsets for comprehensive evaluation
DATASET_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
TEST_SPLIT = "test"
N_FEWSHOT = 4
STOP_SEQUENCE = "Problem:"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8


@dataclass
class MinervaMathEvaluationConfig(WhiteBoxEvaluationConfig):
    """Evaluation Configuration for the Minerva MATH benchmark."""


@register_evaluation(EvalName.MINERVA_MATH)
class MinervaMathEvaluation(WhiteBoxEvaluation[MinervaMathEvaluationConfig]):
    """Minerva MATH Evaluation class.

    Evaluates a model's ability to solve competition-level mathematics problems.
    Uses exact match with LaTeX answer normalization and sympy equivalence checking.
    """

    name: EvalName = EvalName.MINERVA_MATH
    objective: MetricName = MetricName.MINERVA_MATH_EXACT_MATCH
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on MATH dataset to compute the inferences.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts and responses.
        """
        # Load and combine all subsets
        all_test_docs = []
        for subset in DATASET_SUBSETS:
            dataset = datasets.load_dataset(DATASET_NAME, subset, split=TEST_SPLIT)
            processed = utils.process_docs(dataset)
            all_test_docs.extend([dict(item) for item in processed])  # pyright: ignore[reportArgumentType]

        # Get few-shot examples
        fewshot_examples = utils.list_fewshot_samples()

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, all_test_docs, fewshot_examples),
            error_context="Minerva MATH inference",
        )

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Score model outputs using exact match with LaTeX normalization.

        Args:
            inferences: A polars DataFrame with the prompts to LLM and corresponding responses.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored.
        """
        # Load and combine all subsets to get gold answers
        all_test_docs = []
        for subset in DATASET_SUBSETS:
            dataset = datasets.load_dataset(DATASET_NAME, subset, split=TEST_SPLIT)
            processed = utils.process_docs(dataset)
            all_test_docs.extend([dict(item) for item in processed])  # pyright: ignore[reportArgumentType]

        # Extract and normalize answers from responses
        responses_list = list(inferences[InferenceSchema.response])
        scores: list[float] = []

        for response, doc in tqdm(
            zip(responses_list, all_test_docs, strict=False),
            total=len(responses_list),
            desc="Scoring answers",
        ):
            # Extract answer from model response
            unnormalized_answer = utils.get_unnormalized_answer(response)
            model_answer = utils.normalize_final_answer(unnormalized_answer)

            # Get gold answer
            gold_answer = doc["answer"]

            # Check equivalence
            if utils.is_equiv(model_answer, gold_answer):
                scores.append(1.0)
            else:
                scores.append(0.0)

        # Build scores dataframe
        scores_dict = {
            ScoreSchema.prompt: list(inferences[InferenceSchema.prompt]),
            ScoreSchema.response: responses_list,
            ScoreSchema.score: scores,
        }

        return ScoreSchema.validate(pl.from_dict(data=scores_dict))

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate exact match scores into overall evaluation metrics.

        Args:
            scores: A polars DataFrame with the model inferences scored by evaluator.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the metrics and scores.
        """
        scores_dataframe: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        mean_exact_match: float = float(scores_dataframe[ScoreSchema.score].mean())

        _metrics_dataframe: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [str(MinervaMathEvaluation.objective)],
                EvaluationSchema.metric_value: [mean_exact_match],
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)


def _instantiate_model_and_infer(eval_config, test_docs: list[dict], fewshot_examples: list[dict]) -> pl.DataFrame:
    """Run model on MATH dataset to compute inferences.

    Args:
        eval_config: Evaluation configuration.
        test_docs: Test documents from MATH dataset.
        fewshot_examples: Few-shot examples.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    llm: LLM | None = None
    try:
        llm_kwargs = {
            "model": eval_config.model_checkpoint,
            "tensor_parallel_size": (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
        }

        # Use separate tokenizer if specified
        if eval_config.model_config.tokenizer_checkpoint is not None:
            llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=min(eval_config.model_config.max_generation_length, MAX_NEW_TOKENS),
            stop=[STOP_SEQUENCE],
        )

        # Build prompts with few-shot examples
        prompts: list[str] = []
        for doc in test_docs:
            # Build few-shot prefix
            fewshot_prefix = ""
            for example in fewshot_examples[:N_FEWSHOT]:
                fewshot_prefix += utils.format_fewshot_example(example)

            # Build task prompt
            task_prompt = utils.doc_to_text(doc)

            # Combine
            full_prompt = fewshot_prefix + task_prompt
            prompts.append(full_prompt)

        # Run inference
        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        request_outputs = llm.generate(prompts, sampling_params)

        for prompt, request_output in zip(prompts, request_outputs, strict=False):
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
