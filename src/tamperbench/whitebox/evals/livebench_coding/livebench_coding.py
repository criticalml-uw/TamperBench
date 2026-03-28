"""LiveBench Coding evaluation (LCB_generation + coding_completion tasks).

Evaluates a model's ability to generate code solutions for programming
problems from the LiveBench benchmark. Uses pass@1 with actual code
execution via vendored LiveBench evaluation infrastructure.

LiveBench: https://livebench.ai/
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportArgumentType=false, reportCallIssue=false, reportOptionalMemberAccess=false
# ruff: noqa: B905

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import pickle
import zlib
from dataclasses import dataclass

import datasets
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from typing_extensions import override
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from tamperbench.whitebox.evals.livebench_coding._vendor.compute_code_generation_metrics import codegen_metrics
from tamperbench.whitebox.evals.livebench_coding._vendor.extraction_utils import extract_code
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

multiprocessing.set_start_method("spawn", force=True)

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DATASET_PATH = "livebench/coding"
TEST_SPLIT = "test"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8

INCLUDED_TASKS = {"LCB_generation", "coding_completion"}


def _filter_questions(dataset: datasets.Dataset, livebench_release: str, max_samples: int | None) -> list[dict]:
    """Filter dataset questions by release date, removal date, and task type."""
    questions = []
    for item in dataset:
        q = dict(item)

        # Filter by task type
        task = q.get("task") or q.get("subtask", "")
        if task not in INCLUDED_TASKS:
            continue

        # Filter by release date: include questions from releases <= config date
        release_date = q.get("livebench_release_date", "")
        if not release_date or release_date > livebench_release:
            continue

        # Filter by removal date: exclude removed questions
        removal_date = q.get("livebench_removal_date", "")
        if removal_date and removal_date <= livebench_release:
            continue

        questions.append(q)

    if max_samples is not None:
        questions = questions[:max_samples]

    return questions


@dataclass
class LiveBenchCodingEvaluationConfig(WhiteBoxEvaluationConfig):
    """Configuration for LiveBench Coding evaluation."""

    livebench_release: str = "2025-01-08"  # Pin to this release date
    max_samples: int | None = None
    timeout: int = 6  # Per-test timeout (LiveBench default)


@register_evaluation(EvalName.LIVEBENCH_CODING, LiveBenchCodingEvaluationConfig)
class LiveBenchCodingEvaluation(WhiteBoxEvaluation[LiveBenchCodingEvaluationConfig]):
    """LiveBench Coding Evaluation.

    Evaluates a model's ability to solve programming problems from LiveBench's
    LCB_generation and coding_completion tasks. Uses pass@1 with code execution.
    """

    name: EvalName = EvalName.LIVEBENCH_CODING
    objective: MetricName = MetricName.LIVEBENCH_CODING_PASS_AT_1
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Generate code solutions for LiveBench coding problems."""
        dataset = datasets.load_dataset(DATASET_PATH, split=TEST_SPLIT)
        questions = _filter_questions(dataset, self.eval_config.livebench_release, self.eval_config.max_samples)

        print(f"LiveBench Coding: {len(questions)} questions after filtering")

        # Apply chat template to each question
        model_config = self.eval_config.model_config
        prompts = []
        for q in questions:
            question_text = q["turns"][0]
            prompt = f"{model_config.user_prefix}{question_text}{model_config.end_turn}{model_config.assistant_prefix}"
            prompts.append(prompt)

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, prompts),
            kwargs={
                "temperature": TEMPERATURE,
                "max_tokens": min(self.eval_config.model_config.max_generation_length, MAX_NEW_TOKENS),
            },
            error_context="LiveBench Coding inference",
        )

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Execute generated code against test cases and compute pass/fail scores."""
        dataset = datasets.load_dataset(DATASET_PATH, split=TEST_SPLIT)
        questions = _filter_questions(dataset, self.eval_config.livebench_release, self.eval_config.max_samples)

        # Truncate to match inferences length (following MBPP pattern)
        questions = questions[: len(inferences)]

        responses_list = list(inferences[InferenceSchema.response])
        scores = []

        for q, response in zip(questions, responses_list):
            score = _score_single_question(q, response, timeout=self.eval_config.timeout)
            scores.append(score)

        scores_df = pl.DataFrame(
            {
                ScoreSchema.prompt: list(inferences[InferenceSchema.prompt]),
                ScoreSchema.response: responses_list,
                ScoreSchema.score: scores,
            }
        )

        return ScoreSchema.validate(scores_df)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Compute final pass@1 metric."""
        scores_dataframe: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        mean_pass_at_1: float = float(scores_dataframe[ScoreSchema.score].mean())

        print(f"LiveBench Coding pass@1: {mean_pass_at_1:.3f}")

        _metrics_dataframe: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [str(LiveBenchCodingEvaluation.objective)],
                EvaluationSchema.metric_value: [mean_pass_at_1],
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)


def _score_single_question(question: dict, llm_answer: str, timeout: int) -> float:
    """Score a single question following LiveBench's LCB_generation_process_results logic."""
    # Extract code from markdown blocks
    extracted_answer = extract_code(model_output=llm_answer, lmstyle=None)

    # Prepend partial_solution if present and not already a prefix
    partial = question.get("partial_solution")
    if partial and len(partial) > 0 and not extracted_answer.startswith(partial):
        full_solution = partial + "\n" + extracted_answer
    else:
        full_solution = extracted_answer

    # Parse public test cases
    public_test_cases = json.loads(question["public_test_cases"])

    # Parse private test cases (with zlib fallback for compressed format)
    try:
        private_test_cases = json.loads(question["private_test_cases"])
    except (json.JSONDecodeError, TypeError):
        private_test_cases = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(question["private_test_cases"].encode("utf-8"))))
        )

    # Parse metadata for function name
    metadata = json.loads(question["original_json"]["metadata"])

    # Build eval sample
    all_tests = public_test_cases + private_test_cases
    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": [t["input"] for t in all_tests],
                "outputs": [t["output"] for t in all_tests],
                "fn_name": metadata.get("func_name", None),
            }
        )
    }

    # Run evaluation
    metrics, _, _ = codegen_metrics(
        [eval_sample],
        [[full_solution]],
        k_list=[1],
        num_process_evaluate=1,
        timeout=timeout,
    )

    if metrics["pass@1"] == 1.0:
        return 1.0
    return 0.0


def _instantiate_model_and_infer(
    eval_config: LiveBenchCodingEvaluationConfig,
    prompts: list[str],
    *,
    temperature: float,
    max_tokens: int,
) -> pl.DataFrame:
    """Run inference in a subprocess to properly release GPU memory."""
    llm: LLM | None = None
    try:
        llm_kwargs = {
            "model": eval_config.model_checkpoint,
            "tensor_parallel_size": (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
        }

        if eval_config.model_config.tokenizer_checkpoint is not None:
            llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        request_outputs = llm.generate(prompts, sampling_params)

        for prompt, request_output in zip(prompts, request_outputs, strict=False):
            text: str = ""
            if request_output.outputs:
                text = request_output.outputs[0].text

            inferences[InferenceSchema.prompt].append(prompt)
            inferences[InferenceSchema.response].append(text)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
